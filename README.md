# Script Switch Classifier

Lightweight binary script classification for Hindi/Hinglish voice transcripts.

- `DEVANAGARI`: Devanagari letters are the dominant alphabetic script.
- `LATIN`: Latin letters are the dominant alphabetic script.

This classifies **writing script**, not language. Hindi written in Latin letters is Latin-script text.

## Evaluation policy

The supplied `call_data(3).txt` must remain a held-out real-world evaluation set. Do not use it for training, synthetic-data generation, model selection, or threshold tuning. Do not commit raw call transcripts or personal/customer data.

## Initial workflow

1. Inspect public dataset schemas and licensing metadata.
2. Audit the selected model tokenizer on Devanagari, Latin, and mixed examples.
3. Prepare labels using a documented Unicode-letter-count rule.
4. Split reproducibly and prevent duplicate/parallel sentence leakage.
5. Fine-tune only after the tokenizer audit passes.
6. Evaluate once on the untouched call transcript data.

## Selected initial checkpoint

`google/bert_uncased_L-2_H-128_A-2` (BERT-Tiny).

**Important:** this checkpoint is English-oriented. If Devanagari characters are mostly mapped to unknown tokens, do not train it for this task; select a script-capable tokenizer/encoder instead.

## Public datasets to verify

- `LingoIITGN/COMI-LINGUA`
- `debajyotimaz/trilingual-hinglish-corpus`

Check each dataset's current card, licence, splits, schema, and terms before use. Dataset identifiers and schemas may change; the inspection script reports what's available.

## Quick start

Python 3.10–3.12 recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.inspect_datasets
python -m src.data.audit_tokenizer
pytest -q
```

Training scripts should be added after dataset schemas and tokenizer behavior have been verified.

## Label policy

Count alphabetic characters by Unicode script:

- More Devanagari letters -> `DEVANAGARI`
- More Latin letters -> `LATIN`
- Equal counts -> `TIE` (exclude from supervised training by default)
- Other scripts without Devanagari/Latin letters -> `OUT_OF_SCOPE`
- No letters -> `NO_LETTERS`

Digits, whitespace, punctuation, and symbols do not count.

## Privacy and reproducibility

- Never commit credentials, API tokens, phone numbers, or raw call transcripts.
- Store downloaded datasets outside version control.
- Record dataset versions and licences in experiment notes.
- Do not claim an accuracy target is met until measured on held-out data.
