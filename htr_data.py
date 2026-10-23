"""
htr_data.py  --  everything that does NOT need PyTorch.

  * text cleaning, vocabulary, CER / WER
  * dataset discovery + writer-disjoint split
  * PAGE PREPROCESSING  (phone photo of a paragraph -> list of text-line images)
  * CTC decoders: greedy, prefix beam search (+ optional character LM)
  * character n-gram language model
  * Hindi / Marathi language identification (character n-grams + logistic regression)

Why the page preprocessing exists
---------------------------------
The dataset images are ~3000x4000 phone photos of a whole handwritten paragraph
(5-9 lines, margins, shadows, sometimes a cloth background). A CRNN reads ONE line.
So we: fix lighting -> find the paper -> deskew -> split into lines -> resize every
line to the same height -> stitch the lines left-to-right into one long strip.
The strip is trained against the paragraph transcript (lines in reading order).
"""
import json, math, random, re, unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

BLANK = 0                      # CTC blank index (always 0)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------
# 1. text, vocabulary, metrics
# ----------------------------------------------------------------------------
def clean_text(t: str) -> str:
    """NFC, drop invisible joiners, collapse whitespace."""
    t = unicodedata.normalize("NFC", str(t))
    t = t.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def build_vocab(train_texts):
    """Vocabulary from TRAIN text only + the whole Devanagari block (so a rare
    character in val/test is still a legal class). Index 0 is the CTC blank."""
    chars = set("".join(train_texts))
    chars |= {chr(c) for c in range(0x0900, 0x0980)}
    chars.add(" ")
    chars = sorted(chars)
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def encode(text, char2idx):
    """Characters not in the vocabulary are skipped (only possible for val/test)."""
    return [char2idx[c] for c in text if c in char2idx]


def ctc_required_steps(ids):
    """CTC needs one extra step between two identical neighbouring labels."""
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(1, len(r))


def corpus_cer(refs, hyps):
    """Micro-averaged CER: total edits / total reference characters."""
    e = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    return e / max(1, sum(len(r) for r in refs))


def corpus_wer(refs, hyps):
    e = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    return e / max(1, sum(len(r.split()) for r in refs))


# ----------------------------------------------------------------------------
# 2. dataset discovery + writer-disjoint split
# ----------------------------------------------------------------------------
def find_language_folders(root):
    """Return {language: {'val': Path, 'test': Path or None}}.
    Tolerates the 'Marati' spelling of the folder. Only folders that contain .txt
    labels count as labelled."""
    root = Path(root)
    out = {}
    for lang, key in [("Hindi", "hindi"), ("Marathi", "marat")]:
        lang_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith(key)]
        info = {"labelled": [], "unlabelled": []}
        for ld in lang_dirs:
            for sub in [ld] + [s for s in ld.rglob("*") if s.is_dir()]:
                imgs = [p for p in sub.glob("*") if p.suffix.lower() in IMG_EXT]
                txts = list(sub.glob("*.txt"))
                if imgs and txts:
                    info["labelled"].append(sub)
                elif imgs:
                    info["unlabelled"].append(sub)
        out[lang] = info
    return out


def writer_of(stem: str) -> str:
    """File names look like 101_5708 -> writer id is the part before '_'."""
    return stem.split("_")[0]


def build_manifest(root):
    """One row per labelled page: id, language, writer, image, text."""
    import pandas as pd
    folders = find_language_folders(root)
    rows = []
    for lang, info in folders.items():
        for d in info["labelled"]:
            imgs = {p.stem: p for p in d.glob("*") if p.suffix.lower() in IMG_EXT}
            for txt in sorted(d.glob("*.txt")):
                if txt.stem not in imgs:
                    continue
                try:
                    t = txt.read_text(encoding="utf-8-sig")
                except UnicodeDecodeError:
                    t = txt.read_text(encoding="utf-8", errors="ignore")
                t = clean_text(t)
                if t:
                    rows.append(dict(id=f"{lang[:2].lower()}_{txt.stem}", language=lang,
                                     writer=f"{lang[:2].lower()}_{writer_of(txt.stem)}",
                                     image=str(imgs[txt.stem].resolve()), text=t))
    return pd.DataFrame(rows)


def split_by_writer(df, seed=42, fractions=(0.70, 0.15, 0.15), min_writers=12):
    """Assign whole WRITERS to train/val/test (per language) so the same hand never
    appears in two splits. Falls back to a random page split if there are too few writers."""
    rng = random.Random(seed)
    split = {}
    notes = []
    for lang, g in df.groupby("language"):
        writers = sorted(g.writer.unique())
        if len(writers) < min_writers:
            ids = sorted(g.id)
            rng.shuffle(ids)
            n = len(ids)
            a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
            for i, k in enumerate(ids):
                split[k] = "train" if i < a else "val" if i < b else "test"
            notes.append(f"{lang}: only {len(writers)} writers -> random PAGE split")
            continue
        rng.shuffle(writers)
        sizes = g.groupby("writer").size().to_dict()
        total = len(g)
        targets = {"train": fractions[0] * total, "val": fractions[1] * total, "test": fractions[2] * total}
        got = {"train": 0, "val": 0, "test": 0}
        for w in writers:                      # greedy fill: largest remaining deficit first
            s = max(targets, key=lambda k: targets[k] - got[k])
            got[s] += sizes[w]
            for k in g[g.writer == w].id:
                split[k] = s
        notes.append(f"{lang}: {len(writers)} writers -> pages {got}")
    df = df.copy()
    df["split"] = df.id.map(split)
    return df, notes


# ----------------------------------------------------------------------------
# 3. PAGE PREPROCESSING
# ----------------------------------------------------------------------------
def read_gray(path, target_width=2000):
    """Decode a (possibly huge) photo to grayscale, EXIF-rotated, scaled so width <= target_width.
    Scaling by WIDTH (not long side) keeps small handwriting readable when the page only fills
    part of the photo. np.fromfile + imdecode so Windows paths with non-ASCII characters work."""
    data = np.fromfile(str(path), dtype=np.uint8)
    g = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise IOError(f"cannot decode {path}")
    s = min(1.0, target_width / g.shape[1])
    if s < 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return g


def normalize_illumination(g):
    """Divide by an estimate of the paper brightness -> paper ~255, ink dark."""
    k = max(15, int(0.02 * max(g.shape)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)       # removes dark strokes thinner than k
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    return cv2.divide(g, np.maximum(bg, 1), scale=255)


def paper_mask(g):
    """Largest bright region = the sheet of paper (drops cloth / table background).
    Eroded a little so the dark page border and edge shadow are excluded."""
    f = 0.25
    small = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 3)
    thr = 0.6 * np.percentile(blur, 90)
    m = (blur > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones(g.shape, np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    # fill holes (ink, folds): flood-fill the background from the border
    inv = 1 - m
    h, w = inv.shape
    ff = inv.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if ff[seed[1], seed[0]] == 1:
            cv2.floodFill(ff, mask, seed, 2)
    m = (ff != 2).astype(np.uint8)
    e = max(2, int(0.025 * min(h, w)))
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    if m.sum() < 0.15 * m.size:                       # something went wrong -> use everything
        return np.ones(g.shape, np.uint8)
    return cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)


def otsu_threshold(values):
    hist = np.bincount(values.astype(np.uint8).ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        return 128
    sum_all = (np.arange(256) * hist).sum()
    wb = sb = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        wf = total - wb
        if wf == 0:
            break
        sb += t * hist[t]
        mb, mf = sb / wb, (sum_all - sb) / wf
        var = wb * wf * (mb - mf) ** 2
        if var > best:
            best, thr = var, t
    return thr


def ink_mask(norm, paper):
    """Binary ink map inside the paper; removes specks, ruled lines and giant blobs."""
    inside = norm[paper > 0]
    t = int(np.clip(otsu_threshold(inside), 110, 215))
    ink = ((norm < t) & (paper > 0)).astype(np.uint8)
    H, W = ink.shape
    # ruled lines: very long horizontal runs (longer than 40 % of the page width)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.4 * W), 1))
    ruled = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kern)
    ink = cv2.subtract(ink, cv2.dilate(ruled, np.ones((3, 1), np.uint8)))
    # connected-component filter
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    min_area = max(12, int(1e-5 * H * W))
    for i in range(1, n):
        a, h, w = st[i, cv2.CC_STAT_AREA], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_WIDTH]
        if a >= min_area and h < 0.25 * H and w < 0.95 * W:
            keep[i] = True
    return keep[lab].astype(np.uint8)


def estimate_skew(ink, max_angle=10.0):
    """Angle (deg) that makes text rows sharpest = largest variance of the row profile."""
    f = 400.0 / max(ink.shape)
    small = cv2.resize(ink * 255, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h, w = small.shape

    def score(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_LINEAR)
        return float(np.var(r.sum(1)))

    coarse = np.arange(-max_angle, max_angle + 0.01, 1.0)
    a0 = coarse[int(np.argmax([score(a) for a in coarse]))]
    fine = np.arange(a0 - 1.0, a0 + 1.01, 0.2)
    return float(fine[int(np.argmax([score(a) for a in fine]))])


def rotate(img, angle, border):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _runs(bool_arr):
    runs, start = [], None
    for i, v in enumerate(bool_arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(bool_arr)])
    return runs


def segment_lines(ink):
    """Row-profile line segmentation. Returns [(y0, y1), ...] top to bottom."""
    H = ink.shape[0]
    prof = ink.sum(1).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), max(1.5, 0.003 * H)).ravel()
    nz = prof[prof > 0]
    if nz.size == 0:
        return []
    thr = 0.10 * np.percentile(nz, 90)
    runs = [r for r in _runs(prof > thr) if r[1] - r[0] >= max(6, 0.008 * H)]

    def solid(r):                              # solid blob (cloth pattern, finger) -> not text
        cols = np.where(ink[r[0]:r[1]].sum(0) > 0)[0]
        return cols.size == 0 or ink[r[0]:r[1], cols[0]:cols[-1] + 1].mean() > 0.40

    runs = [r for r in runs if not solid(r)]
    if not runs:
        return []
    heights = np.array([r[1] - r[0] for r in runs])
    med = float(np.median(heights))

    # split merged lines (run much taller than the median) at profile valleys
    out = []
    for y0, y1 in runs:
        h = y1 - y0
        n = int(round(h / med))
        if h > 1.6 * med and n >= 2:
            cuts, last = [], y0
            for k in range(1, n):
                centre = y0 + h * k / n
                lo, hi = int(centre - 0.35 * med), int(centre + 0.35 * med)
                lo, hi = max(lo, last + 1), min(hi, y1 - 1)
                if hi <= lo:
                    continue
                c = lo + int(np.argmin(prof[lo:hi]))
                cuts.append(c)
                last = c
            edges = [y0] + cuts + [y1]
            out += [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]
        else:
            out.append([y0, y1])
    runs = out

    # merge / drop very small runs (stray diacritics, specks)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        hs = [r[1] - r[0] for r in runs]
        med = float(np.median(hs))
        for i, r in enumerate(runs):
            if r[1] - r[0] < 0.45 * med:
                gaps = []
                if i > 0:
                    gaps.append((r[0] - runs[i - 1][1], i - 1))
                if i < len(runs) - 1:
                    gaps.append((runs[i + 1][0] - r[1], i + 1))
                g, j = min(gaps)
                if g < 0.6 * med:
                    runs[j] = [min(runs[j][0], r[0]), max(runs[j][1], r[1])]
                runs.pop(i)
                changed = True
                break
    # keep only the main text block: split where the vertical gap is > 3x the line pitch
    # and keep the group with the most lines (drops cloth / table artefacts far from the text)
    if len(runs) >= 4:
        c = np.array([(a + b) / 2 for a, b in runs])
        d = np.diff(c)
        pitch = float(np.median(d))
        groups, cur = [], [0]
        for i, gap in enumerate(d, start=1):
            if gap > 3.0 * pitch:
                groups.append(cur)
                cur = []
            cur.append(i)
        groups.append(cur)
        main = max(groups, key=len)
        runs = [runs[i] for i in main]
    return [(int(a), int(b)) for a, b in runs]


def extract_lines(norm, ink, boxes, cache_h=64, margin_frac=0.18):
    """Cut every line out, clean background -> white, ink contrast-stretched, height = cache_h."""
    H, W = ink.shape
    near_ink = cv2.dilate(ink, np.ones((3, 3), np.uint8)).astype(bool)
    ink_vals = norm[ink > 0]
    lo = float(np.percentile(ink_vals, 5)) if ink_vals.size else 60.0
    gain = 235.0 / max(255.0 - lo, 30.0)
    clean = np.where(near_ink, norm, 255).astype(np.float32)
    clean = np.clip((clean - lo) * gain + 20.0, 0, 255)
    clean[~near_ink] = 255
    clean = clean.astype(np.uint8)

    lines, widths, heights = [], [], []
    for y0, y1 in boxes:
        m = int(margin_frac * (y1 - y0))
        a, b = max(0, y0 - m), min(H, y1 + m)
        cols = np.where(ink[y0:y1].sum(0) > 0)[0]
        if cols.size == 0:
            continue
        density = ink[y0:y1, cols[0]:cols[-1] + 1].mean()
        if density > 0.40:          # solid blob (cloth pattern, finger, logo) - not handwriting
            continue
        x0, x1 = max(0, cols[0] - m), min(W, cols[-1] + m + 1)
        crop = clean[a:b, x0:x1]
        h, w = crop.shape
        nw = max(8, int(round(w * cache_h / h)))
        interp = cv2.INTER_AREA if cache_h < h else cv2.INTER_CUBIC
        lines.append(cv2.resize(crop, (nw, cache_h), interpolation=interp))
        widths.append(int(cols[-1] - cols[0] + 1))
        heights.append(y1 - y0)
    return lines, widths, heights


def drop_non_text_lines(lines, widths, heights, heading_mode="auto"):
    """Heading / page-id line at the top and tiny watermark at the bottom.
    heading_mode: 'auto' (drop a clearly shorter first line), 'drop', 'keep'."""
    info = {"dropped_heading": False, "dropped_footer": False}
    if len(lines) >= 4:
        med_w = float(np.median(widths[1:]))
        if heading_mode == "drop" or (heading_mode == "auto" and widths[0] < 0.6 * med_w):
            lines, widths, heights = lines[1:], widths[1:], heights[1:]
            info["dropped_heading"] = True
    if len(lines) >= 4:
        med_w, med_h = float(np.median(widths[:-1])), float(np.median(heights[:-1]))
        if widths[-1] < 0.18 * med_w and heights[-1] < 0.7 * med_h:
            lines, widths, heights = lines[:-1], widths[:-1], heights[:-1]
            info["dropped_footer"] = True
    return lines, info


def process_page(path, target_width=2000, cache_h=64, heading_mode="auto", return_debug=False):
    """photo -> list of line images (uint8, dark ink on white, height cache_h)."""
    g = read_gray(path, target_width)
    norm = normalize_illumination(g)
    paper = paper_mask(g)
    ink = ink_mask(norm, paper)
    ang = estimate_skew(ink)
    if abs(ang) > 0.1:
        norm = rotate(norm, ang, 255)
        ink = (rotate(ink * 255, ang, 0) > 127).astype(np.uint8)
    boxes = segment_lines(ink)
    lines, widths, heights = extract_lines(norm, ink, boxes, cache_h)
    lines, info = drop_non_text_lines(lines, widths, heights, heading_mode)
    info.update(angle=ang, n_boxes=len(boxes), n_lines=len(lines))
    if return_debug:
        return lines, info, dict(gray=g, norm=norm, paper=paper, ink=ink, boxes=boxes)
    return lines, info


def stitch_lines(lines, height, gap):
    """Resize each line to `height` (keeping aspect) and join them left->right with white gaps."""
    parts = []
    for i, l in enumerate(lines):
        h, w = l.shape
        nw = max(8, int(round(w * height / h)))
        interp = cv2.INTER_AREA if height < h else cv2.INTER_CUBIC
        parts.append(cv2.resize(l, (nw, height), interpolation=interp))
        if i < len(lines) - 1:
            parts.append(np.full((height, gap), 255, np.uint8))
    return np.hstack(parts) if parts else np.full((height, 32), 255, np.uint8)


# ----------------------------------------------------------------------------
# 4. training-time augmentation (applied to lines BEFORE stitching)
# ----------------------------------------------------------------------------
def augment_line(l, height, rng):
    h, w = l.shape
    s = rng.uniform(0.85, 1.15)                       # vertical scale
    sx = rng.uniform(0.85, 1.15)                      # extra horizontal stretch
    th = max(16, int(round(height * s)))
    nw = max(8, int(round(w * th / h * sx)))
    l = cv2.resize(l, (nw, th), interpolation=cv2.INTER_AREA if th < h else cv2.INTER_CUBIC)
    # small rotation + shear
    ang, shear = rng.uniform(-2.0, 2.0), rng.uniform(-0.12, 0.12)
    M = cv2.getRotationMatrix2D((nw / 2, th / 2), ang, 1.0)
    M[0, 1] += shear
    l = cv2.warpAffine(l, M, (nw, th), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # stroke thickness
    r = rng.random()
    if r < 0.2:
        l = cv2.erode(l, np.ones((2, 2), np.uint8))       # erode on white bg = thicker ink
    elif r < 0.35:
        l = cv2.dilate(l, np.ones((2, 2), np.uint8))      # thinner ink
    if rng.random() < 0.3:
        l = cv2.GaussianBlur(l, (0, 0), rng.uniform(0.4, 1.0))
    # contrast / brightness / noise
    f = l.astype(np.float32)
    f = 255 - (255 - f) * rng.uniform(0.75, 1.1)
    f += rng.normal(0, rng.uniform(0, 4), f.shape)
    l = np.clip(f, 0, 255).astype(np.uint8)
    # back to exact height (pad with white / centre-crop)
    if th < height:
        top = (height - th) // 2
        l = np.vstack([np.full((top, nw), 255, np.uint8), l,
                       np.full((height - th - top, nw), 255, np.uint8)])
    elif th > height:
        top = (th - height) // 2
        l = l[top:top + height]
    return l


# ----------------------------------------------------------------------------
# 5. CTC decoding
# ----------------------------------------------------------------------------
def greedy_decode(logp, idx2char, blank=BLANK):
    """logp: numpy [T, V] (log-probabilities or logits)."""
    best = logp.argmax(1)
    out, prev = [], -1
    for i in best:
        if i != blank and i != prev:
            out.append(int(i))
        prev = i
    return "".join(idx2char[i] for i in out)


def _lse(*xs):
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


class CharNgramLM:
    """Character n-gram LM with add-k smoothing, trained on TRAIN transcripts only."""

    def __init__(self, order=4, k=0.05):
        self.n, self.k = order, k
        self.counts = defaultdict(lambda: defaultdict(int))
        self.ctx_total = defaultdict(int)
        self.vocab = set()
        self._cache = {}

    def fit(self, texts):
        for t in texts:
            s = "^" * (self.n - 1) + t + "$"
            for i in range(self.n - 1, len(s)):
                ctx, c = s[i - self.n + 1:i], s[i]
                self.counts[ctx][c] += 1
                self.ctx_total[ctx] += 1
                self.vocab.add(c)
        self._cache = {}
        return self

    def logp(self, history, c):
        ctx = ("^" * (self.n - 1) + history)[-(self.n - 1):] if self.n > 1 else ""
        key = (ctx, c)
        v = self._cache.get(key)
        if v is None:
            V = len(self.vocab) + 1
            v = math.log((self.counts[ctx][c] + self.k) / (self.ctx_total[ctx] + self.k * V))
            self._cache[key] = v
        return v


def prefix_beam_search(logp, idx2char, beam_width=10, blank=BLANK, lm=None, alpha=0.0,
                       beta=0.0, topk=8, cutoff=-12.0):
    """CTC prefix beam search (Hannun et al. 2014).

    logp : numpy [T, V] log-probabilities.
    Every prefix carries two scores: p_b (alignments ending in blank) and p_nb (ending in
    its last character). Merging both is what makes this different from greedy decoding.
    lm / alpha / beta : optional shallow fusion  score += alpha*logP_lm(c|prefix) + beta
    topk / cutoff : only characters among the top-k with logp > cutoff are expanded
                    (set topk=V and cutoff=-inf for the exact, unpruned search).
    """
    NEG = -math.inf
    T, V = logp.shape
    beams = {(): (0.0, NEG)}                      # prefix(tuple of ids) -> (p_b, p_nb)
    text_of = {(): ""}
    for t in range(T):
        row = logp[t]
        cand = [c for c in np.argsort(row)[::-1][:topk] if c != blank and row[c] > cutoff]
        nxt = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            tot = _lse(pb, pnb)
            # 1) blank: prefix unchanged, now ends in blank
            nxt[prefix][0] = _lse(nxt[prefix][0], tot + row[blank])
            # 2) repeat last char with no blank in between: prefix unchanged
            if prefix:
                nxt[prefix][1] = _lse(nxt[prefix][1], pnb + row[prefix[-1]])
            # 3) extend with a character
            for c in cand:
                c = int(c)
                new = prefix + (c,)
                if new not in text_of:
                    text_of[new] = text_of[prefix] + idx2char[c]
                bonus = 0.0
                if lm is not None:
                    bonus = alpha * lm.logp(text_of[prefix], idx2char[c]) + beta
                if prefix and c == prefix[-1]:
                    add = pb + row[c] + bonus               # a repeat needs a blank before it
                else:
                    add = tot + row[c] + bonus
                nxt[new][1] = _lse(nxt[new][1], add)
        scored = sorted(nxt.items(), key=lambda kv: _lse(*kv[1]), reverse=True)[:beam_width]
        beams = {k: (v[0], v[1]) for k, v in scored}
    best = max(beams.items(), key=lambda kv: _lse(*kv[1]))[0]
    return text_of[best]


# ----------------------------------------------------------------------------
# 6. Hindi / Marathi language identification (text based)
# ----------------------------------------------------------------------------
DEVA = [chr(c) for c in range(0x0915, 0x0940)]


def corrupt_text(t, rate, rng):
    """Mimic OCR errors: random substitute / delete / insert of characters."""
    out = []
    for ch in t:
        r = rng.random()
        if r < rate / 3:
            continue
        elif r < 2 * rate / 3:
            out.append(rng.choice(DEVA))
        elif r < rate:
            out.append(ch)
            out.append(rng.choice(DEVA))
        else:
            out.append(ch)
    return "".join(out)


def make_lid_training_set(texts, labels, seed=42):
    """Original paragraphs + OCR-like noisy copies + short word windows."""
    rng = random.Random(seed)
    X, y = [], []
    for t, l in zip(texts, labels):
        X.append(t); y.append(l)
        for rate in (0.10, 0.25, 0.40):
            X.append(corrupt_text(t, rate, rng)); y.append(l)
        words = t.split()
        for _ in range(3):
            if len(words) > 6:
                n = rng.randint(4, min(14, len(words)))
                s = rng.randint(0, len(words) - n)
                chunk = " ".join(words[s:s + n])
                X.append(corrupt_text(chunk, rng.choice([0.0, 0.15, 0.3]), rng)); y.append(l)
    return X, y


class LanguageIdentifier:
    """TF-IDF over character 1-4 grams + logistic regression. Classes: Hindi / Marathi."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), sublinear_tf=True, min_df=2)
        self.clf = LogisticRegression(C=20.0, max_iter=2000)

    def fit(self, texts, labels, seed=42):
        X, y = make_lid_training_set(texts, labels, seed)
        self.clf.fit(self.vec.fit_transform(X), y)
        return self

    def predict(self, texts):
        return list(self.clf.predict(self.vec.transform(texts)))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts)), list(self.clf.classes_)


# ----------------------------------------------------------------------------
# 7. one-call dataset preparation (used by notebook 00)
# ----------------------------------------------------------------------------
DEFAULT_CFG = dict(H=48, cache_h=64, target_width=2000, heading_mode="auto", seed=42,
                   batch_size=4, num_workers=0)


def prepare_dataset(cfg, workers=4, log=print):
    """labelled pages -> writer-disjoint split -> preprocess + cache lines -> manifest.csv, vocab.json"""
    import pickle
    from concurrent.futures import ThreadPoolExecutor
    out = Path(cfg["out_dir"])
    (out / "cache").mkdir(parents=True, exist_ok=True)
    man = build_manifest(cfg["data_root"])
    if man.empty:
        raise RuntimeError("No labelled image/.txt pairs found under data_root. "
                           "Check DATA_ROOT and the folder names (expected Hindi*/Marat* with *-Val).")
    man, notes = split_by_writer(man, cfg["seed"])
    for n in notes:
        log("split:", n)
    char2idx, _ = build_vocab(man[man.split == "train"].text)

    rows = list(man.itertuples(index=False))

    def work(r):
        try:
            lines, info = process_page(r.image, cfg["target_width"], cfg["cache_h"], cfg["heading_mode"])
        except Exception as e:                       # unreadable image
            return dict(id=r.id, ok=False, err=str(e))
        if not lines:
            return dict(id=r.id, ok=False, err="no text lines found")
        cache = out / "cache" / f"{r.id}.pkl"
        with open(cache, "wb") as f:
            pickle.dump(lines, f)
        strip = stitch_lines(lines, cfg["H"], int(cfg["H"] * 0.6))
        return dict(id=r.id, ok=True, cache=str(cache), strip_w=int(strip.shape[1]),
                    n_lines=info["n_lines"], n_boxes=info["n_boxes"],
                    dropped_heading=info["dropped_heading"], dropped_footer=info["dropped_footer"],
                    angle=round(info["angle"], 2))

    res = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(work, rows), 1):
            res.append(r)
            if i % 50 == 0 or i == len(rows):
                log(f"  preprocessed {i}/{len(rows)} pages")
    st = pd_from_records(res)
    man = man.merge(st, on="id", how="left")
    bad = man[~man.ok.fillna(False).astype(bool)]
    if len(bad):
        log(f"WARNING: {len(bad)} pages failed preprocessing and are dropped, e.g.:")
        log(bad[["id", "err"]].head().to_string())
    man = man[man.ok.fillna(False).astype(bool)].copy()

    man["required"] = [ctc_required_steps(encode(t, char2idx)) for t in man.text]
    man["T"] = man.strip_w // 4
    man["feasible"] = man["T"] >= man["required"]
    man["ratio"] = man.strip_w / man.text.str.len()
    med = float(man.ratio.median())
    man["ratio_ok"] = (man.ratio >= 0.6 * med) & (man.ratio <= 1.6 * med)
    man["train_ok"] = man.feasible & man.ratio_ok
    man = man.drop(columns=["ok", "err"], errors="ignore")
    man.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(dict(char2idx=char2idx, blank=BLANK), f, ensure_ascii=False, indent=1)
    cfg2 = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg2, f, indent=2)
    return man, char2idx


def pd_from_records(records):
    import pandas as pd
    return pd.DataFrame(records)


def qc_image(path, cfg, max_h=900):
    """Page with the detected line boxes drawn on it (for visual quality control)."""
    lines, info, dbg = process_page(path, cfg["target_width"], cfg["cache_h"],
                                    cfg["heading_mode"], return_debug=True)
    vis = cv2.cvtColor(dbg["norm"], cv2.COLOR_GRAY2BGR)
    for y0, y1 in dbg["boxes"]:
        cv2.rectangle(vis, (4, y0), (vis.shape[1] - 4, y1), (0, 160, 0), 4)
    s = max_h / vis.shape[0]
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return vis[:, :, ::-1], lines, info          # RGB for matplotlib


# ----------------------------------------------------------------------------
# 8. error analysis helper
# ----------------------------------------------------------------------------
def align_ops(ref, hyp):
    """Levenshtein alignment -> list of (op, ref_char, hyp_char); op in 'sub','del','ins'."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], "")); i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1])); j -= 1
    return ops[::-1]
