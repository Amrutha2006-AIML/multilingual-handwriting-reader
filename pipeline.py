"""
pipeline.py -- one object that turns a handwriting photo into text, language, reliability and translations.

It re-uses the project's own modules (htr_data, htr_torch, htr_translate) and the files written by the notebooks
in htr_outputs/ (config.json, vocab.json, manifest.csv, models/<tag>.pt, models/lid.pkl, results/final_decoder.json).
Nothing is retrained.

    pipe = Pipeline()                       # loads the OCR model (a few seconds)
    r = pipe.recognise("photo.jpg")         # text + language + reliability + preview images   (fast)
    t = pipe.translate(r["text"], r["language"], "English")   # NLLB-600M (slow on CPU; loaded on first use)

Pipeline(mock=True) skips every model and returns canned output. It exists only to test the web app / UI
without PyTorch and is labelled as such everywhere.
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import htr_data as D
import htr_translate as TR

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "htr_outputs"

# A language-ID score below this means "the text is too damaged to tell Hindi from Marathi reliably".
LID_WARN = 0.90
# Languages offered in the UI (all supported by NLLB-600M; names resolve through htr_translate.LANG).
TARGET_LANGUAGES = ["English", "Hindi", "Marathi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali"]


def _jpeg_b64(img, quality=80, max_w=1400):
    """numpy image (gray or RGB) -> data URI, for showing intermediate pictures in the browser."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _draw_boxes(norm, boxes, max_h=900):
    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
    for y0, y1 in boxes:
        cv2.rectangle(vis, (4, y0), (vis.shape[1] - 4, y1), (0, 160, 0), 4)
    s = max_h / vis.shape[0]
    return cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


class Pipeline:
    def __init__(self, mock: bool = False, log=print):
        self.mock = mock
        self.log = log
        self._lock = threading.Lock()          # one OCR at a time (shared model)
        self._tlock = threading.Lock()         # one translation (or model load) at a time
        self._translators = {}                 # beams -> ParagraphTranslator
        self.nllb_state = "not loaded"         # not loaded | loading | ready | error: <msg>
        self.t_load = 0.0
        t0 = time.time()
        if mock:
            self.decoder = {"tag": "mock", "decoder": "mock", "beam": 0, "alpha": 0, "beta": 0}
            self.baseline = 4.132
            self.lm_rel = None
            self.lid = None
            self.t_load = 0.0
            return

        import htr_torch as T                  # imported here so mock mode works without PyTorch
        self.T = T
        os.chdir(ROOT)                         # htr_torch.load_cfg() uses paths relative to the project folder
        self.cfg = T.load_cfg()
        self.data = T.load_data(self.cfg)
        with open(OUT / "results" / "final_decoder.json", encoding="utf-8") as f:
            self.decoder = json.load(f)

        # Character LM used while decoding (same as notebook 04 / predict.py): trainable pages only.
        use_lm = self.decoder["alpha"] > 0
        lm_dec = D.CharNgramLM(order=5).fit(list(self.data["train"].text)) if use_lm else None
        self.rec = T.Recognizer(self.decoder["tag"], self.data, self.cfg, lm=lm_dec,
                                beam=max(1, self.decoder["beam"]), alpha=self.decoder["alpha"],
                                beta=self.decoder["beta"], use_beam=(self.decoder["decoder"] != "greedy"))
        self.device = str(T.get_device())

        # Reliability scoring (same as notebook 06): LM on ALL training-split transcripts, baseline on clean val text.
        man = self.data["manifest"]
        self.lm_rel = D.CharNgramLM(order=5).fit(list(man[man.split == "train"].text))
        self.baseline = float(np.mean([TR.bits_per_char(t, self.lm_rel) for t in man[man.split == "val"].text]))

        with open(OUT / "models" / "lid.pkl", "rb") as f:
            self.lid = pickle.load(f)
        self.t_load = time.time() - t0
        log(f"[pipeline] OCR model '{self.decoder['tag']}' ready on {self.device} in {self.t_load:.1f}s "
            f"(decoder {self.decoder['decoder']}, beam {self.decoder['beam']}, alpha {self.decoder['alpha']}, "
            f"beta {self.decoder['beta']}); reliability baseline {self.baseline:.3f} bits/char")

    # ------------------------------------------------------------------ recognition
    def recognise(self, image_path) -> dict:
        t0 = time.time()
        if self.mock:
            return self._mock_recognise()

        with self._lock:
            lines, info, dbg = D.process_page(str(image_path), self.cfg["target_width"], self.cfg["cache_h"],
                                              self.cfg["heading_mode"], return_debug=True)
            preview = {"lines": _jpeg_b64(_draw_boxes(dbg["norm"], dbg["boxes"]))}
            if not lines:
                return self._empty(info, preview, "No text lines were found in the photo. Use a flat, well-lit photo "
                                                  "of one handwritten paragraph on plain paper.", time.time() - t0)
            cfg, rec, T = self.cfg, self.rec, self.T
            strip = D.stitch_lines(lines, cfg["H"], int(cfg["H"] * 0.6))
            preview["strip"] = _jpeg_b64(strip)
            import torch
            with torch.no_grad():
                x = torch.from_numpy((255 - strip).astype(np.float32) / 255.0)[None, None].to(rec.dev)
                L = torch.tensor([max(1, x.shape[-1] // rec.model.down)], device=rec.dev)
                lp = rec.model(x, L).float().log_softmax(-1)[0].cpu().numpy()
            if rec.use_beam:
                text = D.prefix_beam_search(lp, self.data["idx2char"], rec.beam, lm=rec.lm, alpha=rec.alpha,
                                            beta=rec.beta)
            else:
                text = D.greedy_decode(lp, self.data["idx2char"])
        return self._describe(text.strip(), info, preview, time.time() - t0)

    def _describe(self, text, info, preview, seconds) -> dict:
        n_lines = int(info.get("n_lines", 0)) if isinstance(info, dict) else 0
        if not text:
            return self._empty(info, preview, "The model could not read any characters from this photo.", seconds)
        proba, classes = self.lid.predict_proba([text])
        k = int(np.argmax(proba[0]))
        language, conf = str(classes[k]), float(proba[0][k])
        rel = TR.reliability(text, self.lm_rel, self.baseline)
        warnings = []
        if rel["reliability"] == "LOW":
            warnings.append("Low reading reliability: much of the text was probably misread, so the language and the "
                            "translation may be wrong. Try a sharper, flatter, better-lit photo.")
        elif rel["reliability"] == "MEDIUM":
            warnings.append("Medium reading reliability: some words are probably misread; check the translation "
                            "against the original.")
        if conf < LID_WARN:
            warnings.append(f"Language-ID score is only {conf:.0%}; the text may be too damaged to tell Hindi from "
                            "Marathi.")
        return {"ok": True, "text": text, "language": language, "language_confidence": conf,
                "reliability": rel["reliability"], "bits_per_char": round(rel["bits_per_char"], 3),
                "baseline_bits_per_char": round(self.baseline, 3),
                "invalid_word_rate": round(rel["invalid_token_rate"], 3), "n_lines": n_lines,
                "warnings": warnings, "preview": preview, "seconds": round(seconds, 2), "mock": False}

    def _empty(self, info, preview, message, seconds) -> dict:
        return {"ok": False, "error": message, "text": "", "language": None, "language_confidence": 0.0,
                "reliability": "LOW", "n_lines": int(info.get("n_lines", 0)) if isinstance(info, dict) else 0,
                "warnings": [message], "preview": preview, "seconds": round(seconds, 2), "mock": self.mock}

    def _mock_recognise(self) -> dict:
        text = "यामुळे आपल्या भावातील तातना उद्योगाला अधिक चालना मिळण्याची शक्यता आहे."
        return {"ok": True, "text": text, "language": "Marathi", "language_confidence": 0.9999,
                "reliability": "HIGH", "bits_per_char": 4.2, "baseline_bits_per_char": 4.132,
                "invalid_word_rate": 0.0, "n_lines": 1,
                "warnings": ["DEMO MODE: this is canned text, not a real recognition result."],
                "preview": {}, "seconds": 0.0, "mock": True}

    # ------------------------------------------------------------------ translation
    def _translator(self, beams: int):
        beams = max(1, int(beams))
        if beams not in self._translators:
            if self.mock:
                backend = TR.make_backend("mock")
            else:
                backend = TR.make_backend("nllb_600M", beams=beams, batch_size=4)
            cache = OUT / "translation" / f"cache_{backend.name}.jsonl"
            self._translators[beams] = self._seeded_translator(backend, cache)
        return self._translators[beams]

    @staticmethod
    def _seeded_translator(backend, cache_path):
        """ParagraphTranslator whose cache is pre-filled with the notebook's translations, minus the looping ones."""
        tr = TR.ParagraphTranslator(backend, cache_path)
        dropped = [k for k, v in tr.cache.items() if TR.is_degenerate(v)]
        for k in dropped:
            del tr.cache[k]
        for k in tr.cache:                     # old entries predate the repetition filter: clean them as well
            tr.cache[k] = TR.squash_repeats(tr.cache[k])
        return tr

    def warm_up_translation(self, beams: int = 1):
        """Load NLLB in the background so the first translation does not wait for the model."""
        if self.mock:
            self.nllb_state = "ready"
            return
        try:
            with self._tlock:
                self.nllb_state = "loading"
                tr = self._translator(beams)
                tr.backend._load()
                self.nllb_state = "ready"
        except Exception as e:                 # missing transformers / no internet on first download ...
            self.nllb_state = f"error: {type(e).__name__}: {str(e)[:200]}"
            self.log(f"[pipeline] translation model not available -> {self.nllb_state}")

    def translate(self, text: str, source_language: str, target_language: str, beams: int = 1) -> dict:
        t0 = time.time()
        src, tgt = TR.to_code(source_language), TR.to_code(target_language)
        if src == tgt:
            return {"ok": True, "target": target_language, "text": text, "same_language": True, "seconds": 0.0}
        with self._tlock:
            tr = self._translator(beams)
            try:
                out = tr(text, src, tgt)
            except Exception as e:
                self.nllb_state = f"error: {type(e).__name__}: {str(e)[:200]}"
                return {"ok": False, "target": target_language, "error": f"{type(e).__name__}: {e}"}
            self.nllb_state = "ready"
        return {"ok": True, "target": target_language, "text": TR.squash_repeats(out), "same_language": False,
                "seconds": round(time.time() - t0, 2)}
