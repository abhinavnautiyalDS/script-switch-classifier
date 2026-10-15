# Script Switch Classifier

Lightweight binary script classification for Hindi/Hinglish voice transcripts.

- `DEVANAGARI`: Devanagari letters are the dominant alphabetic script.
- `LATIN`: Latin letters are the dominant alphabetic script.

This classifies **writing script**, not language. Hindi written in Latin letters is Latin-script text.

## Evaluation policy

The real-world `call_data` transcript file is a held-out final evaluation set. Never use it for training, validation, model selection, hyperparameter tuning, or synthetic-data generation. Do not commit raw call transcripts or personal/customer data.

## Selected initial checkpoint

`google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny; 2 transformer layers, hidden size 128).

**Important tokenizer caveat:** this is an English-only, uncased BERT miniature. The training script prints the observed `[UNK]` token rate before training. If Devanagari/mixed text is largely mapped to unknown tokens, treat results as an experiment only and switch to a script-capable multilingual encoder before production use.

## Dataset preparation

The preparation pipeline uses these public sources:

- `LingoIITGN/COMI-LINGUA`
- `debajyotimaz/trilingual-hinglish-corpus`

Review current dataset cards, licences, splits, schemas, and source terms before use. Labels are derived from Unicode script counts, not from the source's language labels.

Run in Python 3.10–3.12:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.inspect_datasets
python -m src.data.prepare_dataset
python -m src.data.audit_tokenizer
pytest -q
```

This creates `data/processed/train.csv`, `validation.csv`, and `public_test.csv`. The training script reads **only train and validation**.

## Fine-tuning

Recommended: open `notebooks/02_finetune_bert_tiny.ipynb` in Google Colab and select a GPU runtime.

Or, after dataset preparation:

```bash
python -m src.train
```

Training settings: seed 42, max sequence length 128, batch size 32, learning rate 5e-5, up to 5 epochs, early stopping based on validation macro-F1. The best checkpoint and validation metrics are written to `artifacts/bert_tiny_script_classifier/`.

Outputs include the model, `results.json`, validation classification report, and confusion matrix. No public-test or real-world call evaluation is run during fine-tuning.

## Label policy

Count alphabetic characters by Unicode script:

- More Devanagari letters -> `DEVANAGARI`
- More Latin letters -> `LATIN`
- Equal counts -> `TIE` (excluded from supervised training)
- Other scripts without Devanagari/Latin letters -> `OUT_OF_SCOPE`
- No letters -> `NO_LETTERS`

Digits, whitespace, punctuation, and symbols do not count.

## Privacy and reproducibility

- Never commit credentials, API tokens, phone numbers, or raw call transcripts.
- Store downloaded datasets and model artifacts outside version control.
- Record dataset versions and licences in experiment notes.
- Do not claim an accuracy target is met until measured on held-out data.
