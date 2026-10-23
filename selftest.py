"""
selftest.py -- run the REAL model once and check it reproduces the results from your notebooks.

    python selftest.py              # 6 test pages
    python selftest.py --pages 20

It takes the first test-split pages from manifest.csv whose photo still exists at the path stored in the manifest
(your dataset folder), reads each photo with the app's pipeline and compares the character error rate with the
one your notebook 04 recorded for the same page (htr_outputs/results/test_final_predictions.csv).
Same model + same decoder + same preprocessing should give (almost) the same text.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import htr_data as D

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--tolerance", type=float, default=0.05, help="allowed mean |CER difference|")
    args = ap.parse_args()

    from pipeline import Pipeline
    pipe = Pipeline()
    man = pd.read_csv(ROOT / "htr_outputs" / "manifest.csv", encoding="utf-8-sig")
    rec = pd.read_csv(ROOT / "htr_outputs" / "results" / "test_final_predictions.csv", encoding="utf-8-sig").set_index("id")
    test = man[(man.split == "test") & man.image.map(lambda p: Path(p).is_file())]
    if test.empty:
        sys.exit("No test photos found at the paths stored in manifest.csv. Put the dataset where it was during "
                 "training, or add a few photos to samples/ and run the app instead.")
    test = test.groupby("language", group_keys=False).head(max(1, args.pages // 2)).head(args.pages)

    diffs = []
    print(f"{'page':16s} {'lang':8s} {'CER now':>8s} {'CER notebook':>13s}  language  reliability")
    for r in test.itertuples():
        out = pipe.recognise(r.image)
        now = D.cer(r.text, out["text"])
        was = float(rec.loc[r.id, "CER"]) if r.id in rec.index else float("nan")
        diffs.append(abs(now - was))
        print(f"{r.id:16s} {r.language:8s} {now:8.3f} {was:13.3f}  {out['language']}  {out['reliability']}")
    mean_diff = sum(diffs) / len(diffs)
    print(f"\nmean |CER difference| = {mean_diff:.3f} (tolerance {args.tolerance})")
    if mean_diff <= args.tolerance:
        print("PASS: the app reproduces the notebook results.")
    else:
        print("CHECK: results differ from the notebook. Compare Python/OpenCV/PyTorch versions with the training "
              "environment, and make sure htr_outputs/models/tcn.pt and results/final_decoder.json belong together.")
        sys.exit(1)


if __name__ == "__main__":
    main()
