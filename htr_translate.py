"""
htr_translate.py -- translation step for the handwritten-text pipeline.

    photo -> recognised text -> language (Hindi/Marathi) -> THIS FILE -> translation(s)

Backends
    "it2_200M"  IndicTrans2 distilled 200M   (AI4Bharat; fast, good on CPU)         <- default
    "it2_1B"    IndicTrans2 1B               (AI4Bharat; best quality, wants a GPU)
    "nllb_600M" NLLB-200 distilled 600M      (Meta; no login needed, 200 languages)
    "mock"      returns the input with a tag (used only to test the plumbing)

Heavy libraries (torch / transformers / IndicTransToolkit) are imported lazily, so this module can be imported
and the non-model parts used without them.
"""
from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from pathlib import Path

# ----------------------------------------------------------------------------------------------
# Languages (FLORES-200 codes, used by both IndicTrans2 and NLLB)
# ----------------------------------------------------------------------------------------------
LANG = {
    "English": "eng_Latn", "Hindi": "hin_Deva", "Marathi": "mar_Deva", "Telugu": "tel_Telu",
    "Tamil": "tam_Taml", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr", "Punjabi": "pan_Guru", "Odia": "ory_Orya", "Nepali": "npi_Deva",
    "Urdu": "urd_Arab",
}
CODE2NAME = {v: k for k, v in LANG.items()}
ENG = "eng_Latn"


def to_code(x: str) -> str:
    """'Hindi' / 'hindi' / 'hin_Deva' -> 'hin_Deva'."""
    if x in CODE2NAME:
        return x
    for k, v in LANG.items():
        if k.lower() == x.strip().lower():
            return v
    raise ValueError(f"unknown language {x!r}; known: {list(LANG)}")


# ----------------------------------------------------------------------------------------------
# Sentence splitting (the translation models are sentence-level; a page is a paragraph)
# ----------------------------------------------------------------------------------------------
_SPLIT = re.compile(r"(?<=[।॥?!\.])\s+")


def split_sentences(text: str, max_words: int = 40):
    """Split on Devanagari danda / ? / ! / . followed by a space; cut over-long pieces every max_words words
    (handwriting often loses punctuation, and the models degrade on very long inputs)."""
    out = []
    for part in _SPLIT.split(" ".join(str(text).split())):
        part = part.strip()
        if not part:
            continue
        words = part.split()
        for i in range(0, len(words), max_words):
            out.append(" ".join(words[i:i + max_words]))
    return out


# ----------------------------------------------------------------------------------------------
# Recognition reliability (reference-free): how trustworthy is this recognised text?
# ----------------------------------------------------------------------------------------------
def _is_mark(c):
    return unicodedata.category(c) in ("Mn", "Mc")


def _is_vowel_sign(c):
    o = ord(c)
    return 0x093A <= o <= 0x094C or o in (0x0955, 0x0956, 0x0957, 0x0962, 0x0963)


def invalid_token_rate(text: str) -> float:
    """Share of words that cannot occur in Devanagari: starting with a vowel sign/virama/etc., or two vowel signs
    in a row. Typical of badly recognised handwriting ('ा ा ा ा')."""
    toks = text.split()
    bad = 0
    for t in toks:
        core = t.strip(".,;:!?।॥\"'()[]-–—")
        if not core:
            continue
        if _is_mark(core[0]) or any(_is_vowel_sign(a) and _is_vowel_sign(b) for a, b in zip(core, core[1:])):
            bad += 1
    return bad / max(1, len(toks))


def bits_per_char(text: str, lm) -> float:
    """Average bits per character of `text` under the project's character n-gram LM (htr_data.CharNgramLM)."""
    text = text.strip()
    if not text:
        return 8.0
    s = sum(lm.logp(text[:i], c) for i, c in enumerate(text))
    return -s / len(text) / math.log(2)


def reliability(text: str, lm, baseline_bpc: float, medium=0.15, low=0.50, invalid_low=0.08) -> dict:
    """
    Indicative tier for recognised text. `baseline_bpc` = bits/char of CLEAN validation transcripts under the same LM.
    HIGH   : bits/char within `medium` of the clean baseline
    MEDIUM : excess below `low`
    LOW    : excess >= `low`, or >= 8 % invalid words  (translation will probably be unreliable)
    The thresholds are heuristic: check them against your own CER (notebook 06, section 3).
    """
    bpc = bits_per_char(text, lm)
    inv = invalid_token_rate(text)
    excess = bpc - baseline_bpc
    tier = "HIGH" if excess <= medium else ("MEDIUM" if excess < low else "LOW")
    if inv >= invalid_low or not text.strip():
        tier = "LOW"
    return {"bits_per_char": bpc, "invalid_token_rate": inv, "reliability": tier}


# ----------------------------------------------------------------------------------------------
# Metrics (needs `pip install sacrebleu`)
# ----------------------------------------------------------------------------------------------
def chrf(hyp: str, ref: str) -> float:
    """chrF++ (character n-gram F-score incl. word bigrams), 0-100."""
    from sacrebleu.metrics import CHRF
    if not hyp.strip() or not ref.strip():
        return 0.0
    return CHRF(word_order=2).sentence_score(hyp, [ref]).score


def bleu(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import BLEU
    if not hyp.strip() or not ref.strip():
        return 0.0
    return BLEU(effective_order=True).sentence_score(hyp, [ref]).score


def squash_repeats(text: str, max_rep: int = 2, max_ngram: int = 8) -> str:
    """Collapse runaway repetition ('A: A: A: ...', '(Sho) (Sho) ...', 'a x, a x, a x, ...').
    Any word n-gram (n = 1..max_ngram) repeated more than `max_rep` times in a row is cut back to `max_rep` copies."""
    toks = str(text).split()
    for n in range(1, max_ngram + 1):
        out, i = [], 0
        while i < len(toks):
            gram = toks[i:i + n]
            reps, j = 1, i + n
            while len(gram) == n and toks[j:j + n] == gram:
                reps += 1
                j += n
            if reps > 1:
                out.extend(gram * min(reps, max_rep))
                i = j
            else:
                out.append(toks[i])
                i += 1
        toks = out
    return " ".join(toks)


def is_degenerate(text: str) -> bool:
    """True if a translation is mostly repetition (a loop that squash_repeats had to cut a lot)."""
    t = str(text).split()
    return len(t) >= 12 and len(squash_repeats(text).split()) < 0.6 * len(t)


def script_share(text: str, prefix: str = "DEVANAGARI") -> float:
    """Fraction of letters in a given script. For English output this should be ~0 (a model that just copies its
    input would score ~1)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(unicodedata.name(c, "").startswith(prefix) for c in letters) / len(letters)


# ----------------------------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------------------------
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Backend:
    name = "base"

    def direct(self, src: str, tgt: str) -> bool:           # can this backend translate src->tgt in one hop?
        return True

    def translate_batch(self, sentences, src: str, tgt: str):
        raise NotImplementedError


class MockBackend(Backend):
    """No model: tags the text. Only for testing files/plots/Excel output without downloading anything."""
    name = "mock"

    def direct(self, src, tgt):
        return src == ENG or tgt == ENG

    def translate_batch(self, sentences, src, tgt):
        return [f"[{CODE2NAME.get(tgt, tgt)}] {s}" for s in sentences]


class NLLBBackend(Backend):
    """facebook/nllb-200-distilled-600M -- open, no login needed, any pair among 200 languages."""

    def __init__(self, model_id="facebook/nllb-200-distilled-600M", beams=4, batch_size=8, name="nllb_600M"):
        self.model_id, self.beams, self.bs, self.name = model_id, beams, batch_size, name
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.device = _device()
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id).to(self.device).eval()
        return self._tok, self._model

    def translate_batch(self, sentences, src, tgt):
        import torch
        tok, model = self._load()
        tok.src_lang = src
        out = []
        for i in range(0, len(sentences), self.bs):
            enc = tok(sentences[i:i + self.bs], return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(self.device)
            # Guard against the repetition loops NLLB falls into on noisy OCR text: no repeated 3-grams, and an output
            # length limit proportional to the input (a translation is rarely > 1.6x the source tokens).
            n_in = int(enc["input_ids"].shape[1])
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                     num_beams=self.beams, max_new_tokens=min(256, int(1.6 * n_in) + 16),
                                     no_repeat_ngram_size=3)
            out += [squash_repeats(s) for s in tok.batch_decode(gen, skip_special_tokens=True)]
        return out


class IndicTrans2Backend(Backend):
    """
    AI4Bharat IndicTrans2.  Two checkpoints are used: indic->English and English->indic (one hop each; an
    Indic->Indic request is routed Indic->English->Indic by ParagraphTranslator). Set use_indic_indic=True to use the
    direct Indic->Indic checkpoint instead.

    Needs:  pip install "transformers<5" sentencepiece IndicTransToolkit
    and a free Hugging Face account that has accepted the licence on each model page (then `huggingface-cli login`).
    """
    IDS = {
        "200M": {"en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
                 "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
                 "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M"},
        "1B": {"en-indic": "ai4bharat/indictrans2-en-indic-1B",
               "indic-en": "ai4bharat/indictrans2-indic-en-1B",
               "indic-indic": "ai4bharat/indictrans2-indic-indic-1B"},
    }

    def __init__(self, size="200M", beams=4, batch_size=8, use_indic_indic=False, name=None):
        self.size, self.beams, self.bs, self.use_ii = size, beams, batch_size, use_indic_indic
        self.name = name or f"it2_{size}"
        self._models = {}
        self._ip = None

    def direct(self, src, tgt):
        return src == ENG or tgt == ENG or self.use_ii

    def _processor(self):
        if self._ip is None:
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except Exception:
                try:
                    from IndicTransToolkit import IndicProcessor
                except Exception as e:
                    raise ImportError(
                        "IndicTransToolkit is not installed (pip install IndicTransToolkit). On Windows it may need "
                        "'Microsoft C++ Build Tools'; if that is a problem use the NLLB backend instead.") from e
            self._ip = IndicProcessor(inference=True)
        return self._ip

    def _load(self, kind):
        if kind not in self._models:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            mid = self.IDS[self.size][kind]
            self.device = _device()
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(mid, trust_remote_code=True).to(self.device).eval()
            self._models[kind] = (tok, model)
        return self._models[kind]

    def translate_batch(self, sentences, src, tgt):
        import torch
        kind = "indic-en" if tgt == ENG else ("en-indic" if src == ENG else "indic-indic")
        tok, model = self._load(kind)
        ip = self._processor()
        batch = ip.preprocess_batch(list(sentences), src_lang=src, tgt_lang=tgt)
        out = []
        for i in range(0, len(batch), self.bs):
            enc = tok(batch[i:i + self.bs], truncation=True, padding="longest", return_tensors="pt",
                      return_attention_mask=True).to(self.device)
            with torch.no_grad():
                gen = model.generate(**enc, use_cache=True, min_length=0, max_length=256,
                                     num_beams=self.beams, num_return_sequences=1)
            out += tok.batch_decode(gen.detach().cpu().tolist(), skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
        return ip.postprocess_batch(out, lang=tgt)


def make_backend(name: str, beams: int = 4, batch_size: int = 8) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "nllb_600M":
        return NLLBBackend(beams=beams, batch_size=batch_size)
    if name == "it2_200M":
        return IndicTrans2Backend("200M", beams, batch_size)
    if name == "it2_1B":
        return IndicTrans2Backend("1B", beams, batch_size)
    raise ValueError(f"unknown backend {name!r}")


# ----------------------------------------------------------------------------------------------
# Paragraph translator: sentence splitting + pivoting + on-disk cache (so an interrupted run can resume)
# ----------------------------------------------------------------------------------------------
class ParagraphTranslator:
    def __init__(self, backend: Backend, cache_path=None):
        self.backend = backend
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        self.seconds = 0.0
        self.n_sentences = 0
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    k, v = json.loads(line)
                    self.cache[k] = v
                except Exception:
                    pass

    def _key(self, src, tgt, sent):
        return f"{self.backend.name}|{src}|{tgt}|{sent}"

    def _sentences(self, sents, src, tgt):
        if src == tgt:
            return list(sents)
        if not self.backend.direct(src, tgt):                 # pivot through English
            return self._sentences(self._sentences(sents, src, ENG), ENG, tgt)
        keys = [self._key(src, tgt, s) for s in sents]
        todo = [(i, s) for i, (k, s) in enumerate(zip(keys, sents)) if k not in self.cache]
        if todo:
            t0 = time.time()
            outs = self.backend.translate_batch([s for _, s in todo], src, tgt)
            self.seconds += time.time() - t0
            self.n_sentences += len(todo)
            new = {}
            for (i, _), o in zip(todo, outs):
                self.cache[keys[i]] = o
                new[keys[i]] = o
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    for k, v in new.items():
                        f.write(json.dumps([k, v], ensure_ascii=False) + "\n")
        return [self.cache[k] for k in keys]

    def __call__(self, text: str, src: str, tgt: str) -> str:
        src, tgt = to_code(src), to_code(tgt)
        sents = split_sentences(text)
        if not sents:
            return ""
        return " ".join(self._sentences(sents, src, tgt)).strip()
