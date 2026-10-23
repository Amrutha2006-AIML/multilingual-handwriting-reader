# Handwriting reader (prototype)

Upload a photo of Hindi or Marathi handwriting. The app shows the recognised text, the language, how reliable the
reading is, and translations (English, Telugu, Tamil, ...). It wraps the models you already trained. Nothing is retrained.

## Run it (Windows, your Python 3.11 "Translation Env")

    cd <this folder>
    ..\translation_env\Scripts\python.exe server.py

Then open http://127.0.0.1:8000. NLLB-600M is already in your Hugging Face cache from notebook 06, so the first
translation downloads nothing new.

Check once that the app reproduces your notebook results (needs the dataset at the paths stored in manifest.csv):

    ..\translation_env\Scripts\python.exe selftest.py

It reads 6 test photos and compares the error rate with the one notebook 04 recorded for the same pages.
`python tests\test_logic.py` checks the repetition filter and the reliability rating without any model.

## What is in the folder

| file | purpose |
|---|---|
| `server.py` | the web app (Python standard library only) |
| `static/index.html` | the page: upload, result, "How well it works" tab |
| `pipeline.py` | photo -> text -> language -> reliability -> translation |
| `htr_data.py`, `htr_torch.py`, `htr_translate.py` | your project code (`htr_translate.py` has the repetition fix) |
| `htr_outputs/models/tcn.pt`, `lid.pkl` | final reader and language identifier |
| `htr_outputs/results/`, `translation/`, `figures/` | numbers and charts shown on the results tab |
| `samples/` | put demo photos here (JPG/PNG); they appear as buttons on the page |

Good demo photos from your unlabelled test folders: `341_47296`, `304_40105` (Hindi), `850_114250`, `869_111710` (Marathi),
`337_42898` (a badly read page, shows the low-reliability warning), `1305_183069` (nothing read).

## How it behaves

* Reading uses the TCN model with beam search and the character language model (settings from `final_decoder.json`).
* Reliability (high / medium / low) uses the same rule as notebook 06. Low means much of the text is probably wrong,
  so the language and the translation may be wrong too. The margin of each ruled sheet takes that colour.
* Language scores below 90% and photos with no readable text get their own messages instead of an answer.
* Translation runs per language after the reading appears. "Fast" uses 1 beam; "Better quality" uses 4 and is slower on a CPU.
* NLLB sometimes repeats a phrase on badly read text (seen in your `model_comparison_pages.csv`). The app blocks
  repeated phrases, limits output length and cuts any remaining loops.

## Limits (say these in the report)

* About 360 training pages and no pre-training: roughly 1 character in 5 is wrong on unseen pages (CER 0.199).
* Only Hindi and Marathi are recognised. Translation targets are any NLLB language.
* The translation score on the results tab measures how well translation survives reading errors, not human-judged quality.
  `reference_translations.csv` was never filled in, so there are no human-reference scores.
* The app serves one user at a time. It is a prototype, not a deployed service.
