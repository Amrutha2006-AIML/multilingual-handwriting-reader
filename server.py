"""
server.py -- local web app for the handwriting pipeline (standard library only, no Flask / Gradio needed).

    python server.py                 # http://127.0.0.1:8000
    python server.py --port 8080
    python server.py --mock          # UI test without PyTorch / models (canned output, clearly labelled)

API (all JSON)
    GET  /api/status        model + translation-model state
    GET  /api/results       numbers shown on the Results tab
    GET  /api/samples       sample photos found in samples/
    POST /api/recognise     body = raw image bytes          -> text, language, reliability, preview pictures
    POST /api/translate     {"text","source","target","beams"} -> one translation
"""
import argparse
import json
import mimetypes
import os
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
OUT = ROOT / "htr_outputs"
FIGS = OUT / "figures"
SAMPLES = ROOT / "samples"
MAX_UPLOAD = 25 * 1024 * 1024
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

FIGURES = [  # file, caption
    ("model_comparison.png", "Model comparison on the test pages (greedy decoding)"),
    ("training_tcn.png", "TCN (final model): training curves"),
    ("training_cnn.png", "CNN: training curves"),
    ("training_transformer.png", "Transformer: training curves"),
    ("error_analysis.png", "Recognition errors vs. number of detected lines; CER distribution"),
    ("language_id.png", "Language ID: robustness to noise and confusion matrices"),
    ("translation_error_propagation.png", "Translation quality kept vs. recognition error"),
]

PIPE = None


def read_results():
    """Everything the Results tab shows, read from the files the notebooks wrote."""
    import pandas as pd
    res = OUT / "results"
    summary = json.load(open(res / "final_summary.json", encoding="utf-8"))
    comp = pd.read_csv(res / "model_comparison.csv")
    tune = pd.read_csv(res / "decoder_tuning_val.csv")
    out = {
        "models": comp.round(4).to_dict("records"),
        "final": summary["final_decoder"],
        "language_id": {k: {"accuracy": v["accuracy"], "cm": v["cm"]} for k, v in summary["language_id"].items()},
        "decoder_tuning": tune.round(4).head(6).to_dict("records"),
        "figures": [{"file": f, "caption": c} for f, c in FIGURES if (FIGS / f).exists()],
    }
    pages = OUT / "translation" / "model_comparison_pages.csv"
    if pages.exists():
        t = pd.read_csv(pages, encoding="utf-8-sig")
        out["translation"] = {
            "model": "NLLB-200 distilled 600M", "pages": int(len(t)),
            "retention_chrF": round(float(t.retention.mean()), 1),
            "by_reliability": {k: round(float(v), 1) for k, v in t.groupby("reliability").retention.mean().items()},
            "spearman_cer_vs_retention": round(float(t[["CER", "retention"]].corr(method="spearman").iloc[0, 1]), 2),
        }
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "HTRDemo/1.0"

    # ---- helpers
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, root: Path):
        try:
            path = path.resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise FileNotFoundError
        except Exception:
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            raise ValueError("empty request body")
        if n > MAX_UPLOAD:
            raise ValueError(f"upload too large (limit {MAX_UPLOAD // 1024 // 1024} MB)")
        return self.rfile.read(n)

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            print("[http]", fmt % args)

    # ---- routes
    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html", STATIC)
        if path == "/api/status":
            p = PIPE
            return self._json({"mock": p.mock, "ocr_model": p.decoder["tag"], "decoder": p.decoder["decoder"],
                               "beam": p.decoder["beam"], "device": getattr(p, "device", "mock"),
                               "nllb": p.nllb_state, "test_cer": p.decoder.get("test_CER"),
                               "targets": __import__("pipeline").TARGET_LANGUAGES})
        if path == "/api/results":
            try:
                return self._json(read_results())
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        if path == "/api/samples":
            files = sorted(p.name for p in SAMPLES.glob("*") if p.suffix.lower() in IMG_EXT) if SAMPLES.exists() else []
            return self._json({"samples": files})
        if path.startswith("/figures/"):
            return self._file(FIGS / path[len("/figures/"):], FIGS)
        if path.startswith("/samples/"):
            return self._file(SAMPLES / path[len("/samples/"):], SAMPLES)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/recognise":
                data = self._body()
                with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as f:
                    f.write(data)
                    tmp = f.name
                try:
                    import cv2
                    import numpy as np
                    if not PIPE.mock and cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) is None:
                        return self._json({"ok": False, "error": "This file is not a readable image "
                                                                 "(use JPG or PNG)."}, 400)
                    return self._json(PIPE.recognise(tmp))
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            if path == "/api/translate":
                req = json.loads(self._body().decode("utf-8"))
                text, src, tgt = str(req.get("text", "")).strip(), req.get("source"), req.get("target")
                if not text or not src or not tgt:
                    return self._json({"ok": False, "error": "text, source and target are required"}, 400)
                beams = min(4, max(1, int(req.get("beams", 1))))
                return self._json(PIPE.translate(text, src, tgt, beams))
            return self._json({"error": "not found"}, 404)
        except ValueError as e:
            return self._json({"ok": False, "error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global PIPE
    ap = argparse.ArgumentParser(description="Handwriting recognition + language ID + translation web app")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--mock", action="store_true", help="UI test mode without models (canned output)")
    ap.add_argument("--no-warmup", action="store_true", help="do not pre-load the translation model")
    args = ap.parse_args()

    os.chdir(ROOT)
    from pipeline import Pipeline
    print("Loading models ..." if not args.mock else "MOCK MODE: no models are loaded")
    PIPE = Pipeline(mock=args.mock)
    if not args.no_warmup:
        threading.Thread(target=PIPE.warm_up_translation, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  Open  http://{args.host}:{args.port}   (Ctrl+C to stop)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()

