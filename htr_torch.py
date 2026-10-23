"""
htr_torch.py -- PyTorch part: dataset, the three models, training, evaluation, inference.

Models (same CNN backbone, different sequence head):
    "cnn"          CNN backbone + 1-D convolution head          (no recurrence, no attention)
    "tcn"          CNN backbone + 5 dilated residual 1-D conv blocks (wide context, fully parallel, fast)
    "bilstm"       CNN backbone + 2-layer BiLSTM                (CRNN; slow on CPU - not used on a CPU-only PC)
    "transformer"  CNN backbone + Transformer encoder
All three are trained with CTC on the stitched paragraph strip (see htr_data.py).
"""
import json, math, os, pickle, random, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

import htr_data as D

import torch.nn.functional as F
# The backbone shrinks the width by `down` (4 or 8). down=8 halves the number of time steps T, which makes
# the BiLSTM / Transformer (and the CNN) much faster. Each model remembers its own `down`.


# ----------------------------------------------------------------------------
# setup helpers
# ----------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cfg(path="htr_outputs/config.json"):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["out"] = Path(cfg["out_dir"])
    return cfg


def n_classes(data):
    """characters + the CTC blank (index 0)."""
    return len(data["char2idx"]) + 1


def load_data(cfg):
    """manifest + vocabulary written by notebook 00."""
    out = cfg["out"]
    man = pd.read_csv(out / "manifest.csv", encoding="utf-8-sig")
    with open(out / "vocab.json", encoding="utf-8") as f:
        v = json.load(f)
    char2idx = v["char2idx"]
    idx2char = {int(i): c for c, i in char2idx.items()}
    data = dict(char2idx=char2idx, idx2char=idx2char, manifest=man,
                train=man[(man.split == "train") & (man.train_ok)].reset_index(drop=True),
                val=man[man.split == "val"].reset_index(drop=True),
                test=man[man.split == "test"].reset_index(drop=True))
    return data


# ----------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------
class StripDataset(Dataset):
    """One item = one page: its cached text lines are (augmented), resized to height H and
    stitched into one strip. Ink = 1, background = 0 so right-padding with 0 is 'empty paper'."""

    def __init__(self, df, char2idx, H=48, train=False):
        self.df = df.reset_index(drop=True)
        self.char2idx, self.H, self.train = char2idx, H, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with open(row["cache"], "rb") as f:
            lines = pickle.load(f)
        if self.train:
            rng = np.random.default_rng()
            lines = [D.augment_line(l, self.H, rng) for l in lines]
            gap = int(self.H * rng.uniform(0.3, 0.9))
        else:
            gap = int(self.H * 0.6)
        strip = D.stitch_lines(lines, self.H, gap)
        x = torch.from_numpy((255 - strip).astype(np.float32) / 255.0).unsqueeze(0)
        ids = D.encode(row["text"], self.char2idx)
        return dict(x=x, ids=ids, text=row["text"], id=row["id"], language=row["language"])


def collate(batch):
    W = max(b["x"].shape[-1] for b in batch)
    xs = torch.zeros(len(batch), 1, batch[0]["x"].shape[1], W)
    for i, b in enumerate(batch):
        xs[i, :, :, :b["x"].shape[-1]] = b["x"]
    widths = torch.tensor([b["x"].shape[-1] for b in batch], dtype=torch.long)
    tgt = torch.tensor([t for b in batch for t in b["ids"]], dtype=torch.long)
    tlen = torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long)
    return dict(x=xs, widths=widths, targets=tgt, target_lengths=tlen,
                texts=[b["text"] for b in batch], ids=[b["id"] for b in batch],
                languages=[b["language"] for b in batch])


def out_lengths(widths, down):
    """number of CTC time steps of every item = pixel width // down."""
    return (widths // down).clamp(min=1)


def train_pages(data, down):
    """training pages that are usable at this downsampling: CTC feasible and sane strip width."""
    t = data["manifest"]
    t = t[t.split == "train"]
    ok = ((t.strip_w // down) >= t.required) & t.ratio_ok.astype(bool)
    return t[ok].reset_index(drop=True)


def set_threads(n=None):
    """CPU thread count. Small LSTM ops are often FASTER with fewer threads; try 2/4/8 with benchmark()."""
    import os
    n = n or max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(n)
    return n


class BucketSampler(Sampler):
    """Batches pages of similar strip width together (less padding, faster)."""

    def __init__(self, widths, batch_size, shuffle):
        self.w, self.bs, self.shuffle = list(widths), batch_size, shuffle

    def __iter__(self):
        idx = list(range(len(self.w)))
        if self.shuffle:
            random.shuffle(idx)
        chunk = self.bs * 6
        batches = []
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s:s + chunk], key=lambda i: self.w[i])
            batches += [part[k:k + self.bs] for k in range(0, len(part), self.bs)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.w) / self.bs)


def make_loader(df, data, cfg, train, batch_size=None):
    ds = StripDataset(df, data["char2idx"], cfg["H"], train=train)
    bs = batch_size or cfg["batch_size"]
    sampler = BucketSampler(df["strip_w"].tolist(), bs, shuffle=train)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                      num_workers=cfg.get("num_workers", 0))


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def conv_bn(i, o):
    return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class PosEnc(nn.Module):
    def __init__(self, d, max_len=8192):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TCNBlock(nn.Module):
    """Residual block of dilated 1-D convolutions (a 'temporal convolutional network' block).
    Every block sees further along the strip, so the stack covers a wide context like a BiLSTM does,
    but all time steps are computed in parallel (fast on a CPU)."""

    def __init__(self, d, dilation, k=5, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, d, k, padding=dilation * (k - 1) // 2, dilation=dilation),
            nn.BatchNorm1d(d), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(d, d, 1), nn.BatchNorm1d(d))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):                      # x: B, d, T
        return self.act(x + self.net(x))


class HTRNet(nn.Module):
    def __init__(self, head, vocab_size, H=48, down=8, d=256, lstm_hidden=128, tr_layers=3,
                 tr_heads=4, dropout=0.2, chunk=64, ctx=16):
        super().__init__()
        assert H % 16 == 0, "strip height must be a multiple of 16"
        assert down in (4, 8), "down must be 4 or 8"
        self.head_type, self.down, self.chunk, self.ctx = head, down, chunk, ctx
        pool3 = nn.MaxPool2d((2, 2)) if down == 8 else nn.MaxPool2d((2, 1))
        self.backbone = nn.Sequential(
            *conv_bn(1, 32), nn.MaxPool2d(2, 2),
            *conv_bn(32, 64), nn.MaxPool2d(2, 2),
            *conv_bn(64, 128), *conv_bn(128, 128), pool3,
            *conv_bn(128, 256), nn.MaxPool2d((2, 1)))
        feat = 256 * (H // 16)
        self.proj = nn.Sequential(nn.Linear(feat, d), nn.ReLU(inplace=True), nn.Dropout(dropout))
        if head == "cnn":
            self.seq = nn.Sequential(
                nn.Conv1d(d, d, 5, padding=2), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Conv1d(d, d, 5, padding=2), nn.ReLU(inplace=True), nn.Dropout(dropout))
            out_dim = d
        elif head == "tcn":
            self.seq = nn.Sequential(*[TCNBlock(d, dil, dropout=dropout) for dil in (1, 2, 4, 8, 16)])
            out_dim = d
        elif head == "bilstm":
            self.seq = nn.LSTM(d, lstm_hidden, num_layers=2, bidirectional=True,
                               dropout=dropout, batch_first=True)
            out_dim = 2 * lstm_hidden
        elif head == "transformer":
            self.pos = PosEnc(d)
            layer = nn.TransformerEncoderLayer(d, tr_heads, 4 * d, dropout=0.1, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, tr_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            out_dim = d
        else:
            raise ValueError(head)
        self.classifier = nn.Linear(out_dim, vocab_size)

    def _chunked_lstm(self, f):
        """BiLSTM over overlapping windows instead of the whole strip.
        A strip has hundreds of time steps; an LSTM must walk through them one by one (slow on CPU).
        Here the strip is cut into windows of `chunk + 2*ctx` steps (ctx steps of context on both sides),
        all windows run as one batch, and only the central `chunk` steps of every window are kept.
        Sequential depth: ~96 steps instead of ~550-2000."""
        B, T, d = f.shape
        c, ctx = self.chunk, self.ctx
        n = math.ceil(T / c)
        f = F.pad(f, (0, 0, ctx, ctx + n * c - T))                     # B, n*c + 2*ctx, d
        w = f.unfold(1, c + 2 * ctx, c)                                # B, n, d, c + 2*ctx
        w = w.permute(0, 1, 3, 2).reshape(B * n, c + 2 * ctx, d)
        y, _ = self.seq(w)                                             # B*n, c + 2*ctx, 2h
        y = y[:, ctx:ctx + c].reshape(B, n * c, -1)[:, :T]
        return y

    def forward(self, x, lengths):
        f = self.backbone(x)                                  # B, C, h, T
        b, c, h, T = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(b, T, c * h))   # B, T, d
        lengths = lengths.clamp(min=1, max=T)
        if self.head_type in ("cnn", "tcn"):
            y = self.seq(f.transpose(1, 2)).transpose(1, 2)
        elif self.head_type == "bilstm":
            y = self._chunked_lstm(f)
        else:
            pad = torch.arange(T, device=f.device).unsqueeze(0) >= lengths.to(f.device).unsqueeze(1)
            y = self.norm(self.seq(self.pos(f), src_key_padding_mask=pad))
        return self.classifier(y)                             # B, T, V


def build_model(head, vocab_size, cfg, down=8):
    return HTRNet(head, vocab_size, H=cfg["H"], down=down)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_model(model, loader, device, idx2char, ctc=None, keep_logp=False):
    """Greedy-decode every page. Returns (DataFrame, mean CTC loss, [log-prob arrays])."""
    model.eval()
    rows, losses, logps = [], [], []
    for b in loader:
        lens = out_lengths(b["widths"], model.down)
        out = model(b["x"].to(device), lens.to(device)).float()
        lp = out.log_softmax(-1)
        if ctc is not None:
            l = ctc(lp.transpose(0, 1), b["targets"], lens, b["target_lengths"])
            if torch.isfinite(l):
                losses.append(float(l))
        lp_np = lp.cpu().numpy()
        for i in range(len(b["texts"])):
            L = int(lens[i])
            pred = D.greedy_decode(lp_np[i, :L], idx2char)
            rows.append(dict(id=b["ids"][i], language=b["languages"][i],
                             reference=b["texts"][i], prediction=pred))
            if keep_logp:
                logps.append(lp_np[i, :L].copy())
    df = pd.DataFrame(rows)
    df["CER"] = [D.cer(r, p) for r, p in zip(df.reference, df.prediction)]
    df["WER"] = [D.wer(r, p) for r, p in zip(df.reference, df.prediction)]
    return df, (float(np.mean(losses)) if losses else float("nan")), logps


def summarize(df):
    return dict(CER=D.corpus_cer(df.reference, df.prediction),
                WER=D.corpus_wer(df.reference, df.prediction))


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def lr_at(epoch, epochs, lr, warmup=3):
    if epoch < warmup:
        return lr * (epoch + 1) / warmup
    p = (epoch - warmup) / max(1, epochs - warmup)
    return lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def _step(model, b, ctc, dev, opt=None):
    """one forward (+ backward if opt is given) pass; returns the loss tensor."""
    lens = out_lengths(b["widths"], model.down)
    out = model(b["x"].to(dev), lens.to(dev)).float()
    lp = out.log_softmax(-1).transpose(0, 1)                    # T, B, V
    loss = ctc(lp, b["targets"], lens, b["target_lengths"])
    if opt is not None and torch.isfinite(loss):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return loss


def benchmark(head, data, cfg, down=8, n_batches=3, threads=None):
    """Measure the real training speed in about a minute BEFORE starting a long run.
    Prints seconds per batch and the projected minutes per epoch."""
    if threads:
        torch.set_num_threads(threads)
    seed_everything(cfg["seed"])
    dev = get_device()
    tp = train_pages(data, down)
    df = tp.sample(min(len(tp), 4 * (n_batches + 1)), random_state=0)
    loader = make_loader(df, data, cfg, train=True)
    model = build_model(head, n_classes(data), cfg, down).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ctc = nn.CTCLoss(blank=D.BLANK, zero_infinity=True)
    model.train()
    times = []
    for i, b in enumerate(loader):
        t0 = time.time()
        _step(model, b, ctc, dev, opt)
        times.append(time.time() - t0)
        if i + 1 >= n_batches + 1:
            break
    spb = float(np.mean(times[1:] if len(times) > 1 else times))      # first batch = warm-up
    per_epoch = spb * math.ceil(len(tp) / cfg["batch_size"]) / 60
    print(f"{head:11s} down={down} threads={torch.get_num_threads():2d} : {spb:5.1f} s/batch "
          f"-> about {per_epoch:5.1f} min per epoch ({len(tp)} training pages)")
    return spb


def train_model(head, data, cfg, epochs=60, lr=1e-3, patience=15, min_epochs=20, tag=None,
                train_df=None, augment=True, log_every=1, quiet=False, down=8):
    """Train one model. Best checkpoint = lowest validation CER (greedy).
    Patience only starts counting after the model has left the CTC 'all blank' plateau.
    You may interrupt the kernel at any time: the best checkpoint so far is already saved."""
    tag = tag or head
    seed_everything(cfg["seed"])
    dev = get_device()
    tr_df = train_pages(data, down) if train_df is None else train_df
    train_loader = make_loader(tr_df, data, cfg, train=augment)
    val_loader = make_loader(data["val"], data, cfg, train=False, batch_size=4)
    model = build_model(head, n_classes(data), cfg, down).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    ctc = nn.CTCLoss(blank=D.BLANK, zero_infinity=True)
    models_dir = cfg["out"] / "models"; models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / f"{tag}.pt"
    hist, best, best_ep, bad, left_plateau = [], float("inf"), 0, 0, False
    t0 = time.time()
    if not quiet:
        print(f"[{tag}] device={dev} threads={torch.get_num_threads()} down={down} params={count_params(model):,} "
              f"train pages={len(tr_df)} val pages={len(data['val'])}")
    try:
        for ep in range(epochs):
            for g in opt.param_groups:
                g["lr"] = lr_at(ep, epochs, lr)
            model.train()
            tl, n_bad = [], 0
            for b in train_loader:
                loss = _step(model, b, ctc, dev, opt)
                if torch.isfinite(loss):
                    tl.append(loss.item())
                else:
                    n_bad += 1
            val_df, val_loss, _ = run_model(model, val_loader, dev, data["idx2char"], ctc)
            m = summarize(val_df)
            hist.append(dict(epoch=ep + 1, train_loss=float(np.mean(tl)) if tl else float("nan"),
                             val_loss=val_loss, val_CER=m["CER"], val_WER=m["WER"],
                             lr=opt.param_groups[0]["lr"], nonfinite_batches=n_bad))
            if m["CER"] < 0.95:
                left_plateau = True
            if m["CER"] < best:
                best, best_ep, bad = m["CER"], ep + 1, 0
                torch.save(dict(state=model.state_dict(), head=head, H=cfg["H"], down=down,
                                val_CER=best, epoch=ep + 1), ckpt)
            elif left_plateau:
                bad += 1
            if not quiet and ((ep + 1) % log_every == 0 or ep == 0):
                el = time.time() - t0
                print(f"[{tag}] ep {ep+1:3d}/{epochs} | train {hist[-1]['train_loss']:.3f} | "
                      f"val loss {val_loss:.3f} | val CER {m['CER']:.3f} WER {m['WER']:.3f} | "
                      f"best {best:.3f} @ {best_ep} | {el/60:.1f} min")
            if left_plateau and bad >= patience and ep + 1 >= min_epochs:
                if not quiet:
                    print(f"[{tag}] early stop (no val-CER gain for {patience} epochs)")
                break
    except KeyboardInterrupt:
        print(f"[{tag}] interrupted by you - continuing with the best checkpoint so far (epoch {best_ep}).")
    hist_df = pd.DataFrame(hist)
    (cfg["out"] / "results").mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(cfg["out"] / "results" / f"{tag}_history.csv", index=False)
    return dict(model=model, history=hist_df, ckpt=ckpt, best_val_CER=best, best_epoch=best_ep,
                minutes=(time.time() - t0) / 60, device=dev, down=down)


def load_model(ckpt_path, data, cfg):
    dev = get_device()
    ck = torch.load(ckpt_path, map_location=dev)
    model = build_model(ck["head"], n_classes(data), cfg, ck.get("down", 4)).to(dev)   # old checkpoints: down=4
    model.load_state_dict(ck["state"])
    model.eval()
    return model


def evaluate_checkpoint(tag, data, cfg, keep_logp=False):
    """Load the best checkpoint of `tag`; greedy-decode val and test."""
    dev = get_device()
    model = load_model(cfg["out"] / "models" / f"{tag}.pt", data, cfg)
    ctc = nn.CTCLoss(blank=D.BLANK, zero_infinity=True)
    res = {}
    for split in ("val", "test"):
        loader = make_loader(data[split], data, cfg, train=False, batch_size=4)
        df, loss, lps = run_model(model, loader, dev, data["idx2char"], ctc, keep_logp)
        res[split] = dict(df=df, loss=loss, logps=lps, **summarize(df))
    return model, res


def save_result(tag, head, res, train_out, cfg, data):
    r = dict(tag=tag, head=head, down=int(train_out.get("down", 4)), params=count_params(train_out["model"]),
             best_epoch=train_out["best_epoch"], train_minutes=round(train_out["minutes"], 1),
             val_CER=res["val"]["CER"], val_WER=res["val"]["WER"],
             test_CER=res["test"]["CER"], test_WER=res["test"]["WER"],
             train_pages=int(len(train_pages(data, int(train_out.get("down", 4))))),
             val_pages=int(len(data["val"])), test_pages=int(len(data["test"])))
    d = cfg["out"] / "results"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    res["test"]["df"].to_csv(d / f"{tag}_test_predictions.csv", index=False, encoding="utf-8-sig")
    res["val"]["df"].to_csv(d / f"{tag}_val_predictions.csv", index=False, encoding="utf-8-sig")
    return r


def overfit_check(head, data, cfg, n_pages=6, epochs=120, lr=1e-3, target_cer=0.05, down=8):
    """Optional sanity test: can the model memorise a handful of pages (no augmentation)?
    NOTE: n_pages=6 means only 2 optimiser steps per epoch, so this needs hundreds of steps before the
    CTC plateau breaks; a failure here does NOT prove the pipeline is wrong (a normal training run is
    the better test)."""
    seed_everything(cfg["seed"])
    dev = get_device()
    df = train_pages(data, down).head(n_pages).copy()
    loader = make_loader(df, data, cfg, train=False, batch_size=4)
    model = build_model(head, n_classes(data), cfg, down).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=D.BLANK, zero_infinity=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b in loader:
            loss = _step(model, b, ctc, dev, opt)
        if ep % 10 == 0 or ep == 1:
            pdf, _, _ = run_model(model, loader, dev, data["idx2char"])
            c = D.corpus_cer(pdf.reference, pdf.prediction)
            print(f"overfit {head}: epoch {ep:3d} loss {loss.item():.3f}  CER on the {n_pages} training pages {c:.3f}  "
                  f"({(time.time()-t0)/60:.1f} min)")
            if c <= target_cer:
                print("PASS: the pipeline can learn (images and labels are aligned).")
                return True
    print(f"Could not memorise {n_pages} pages in {epochs} epochs (inconclusive - see the docstring).")
    return False


# ----------------------------------------------------------------------------
# inference on a NEW photo
# ----------------------------------------------------------------------------
class Recognizer:
    """photo -> text.   Uses the same preprocessing as training (auto heading rule)."""

    def __init__(self, tag, data, cfg, lm=None, beam=10, alpha=0.0, beta=0.0, use_beam=True):
        self.cfg, self.data = cfg, data
        self.dev = get_device()
        self.model = load_model(cfg["out"] / "models" / f"{tag}.pt", data, cfg)
        self.lm, self.beam, self.alpha, self.beta, self.use_beam = lm, beam, alpha, beta, use_beam

    @torch.no_grad()
    def __call__(self, image_path):
        cfg = self.cfg
        lines, info = D.process_page(image_path, cfg["target_width"], cfg["cache_h"], cfg["heading_mode"])
        if not lines:
            return "", info, None
        strip = D.stitch_lines(lines, cfg["H"], int(cfg["H"] * 0.6))
        x = torch.from_numpy((255 - strip).astype(np.float32) / 255.0)[None, None].to(self.dev)
        L = torch.tensor([max(1, x.shape[-1] // self.model.down)], device=self.dev)
        lp = self.model(x, L).float().log_softmax(-1)[0].cpu().numpy()
        if self.use_beam:
            text = D.prefix_beam_search(lp, self.data["idx2char"], self.beam, lm=self.lm,
                                        alpha=self.alpha, beta=self.beta)
        else:
            text = D.greedy_decode(lp, self.data["idx2char"])
        return text, info, strip
