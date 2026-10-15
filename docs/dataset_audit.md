# Initial dataset audit (public dataset cards)

Audit date: 2026-10-15

This is a metadata/schema review from the public Hugging Face dataset cards, not a local download or execution of the datasets.

## 1. COMI-LINGUA

- Dataset ID: `LingoIITGN/COMI-LINGUA`
- Dataset card: https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA
- Published license: CC BY 4.0
- Available configurations include `LID`, `MLI`, `POS`, `NER`, `TN`, and `MT`.
- The `LID` configuration has train/test splits and includes sentence text plus token-level language-identification annotations. The card's example uses a `Sentences` field and `Predicted tags` / annotator fields.
- Use the raw sentence text to derive our target label using the project's Unicode script-counting rule. Do not mistake token language labels (Hindi/English) for writing-script labels.
- Confirm exact feature names in the installed `datasets` library before writing the loader; dataset cards can evolve.
- Citation: Sheth, Beniwal, and Singh, “COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing,” EMNLP Findings 2025.
- Source: https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA

## 2. Trilingual Hinglish Corpus

- Dataset ID: `debajyotimaz/trilingual-hinglish-corpus`
- Dataset card: https://huggingface.co/datasets/debajyotimaz/trilingual-hinglish-corpus
- Published license: CC BY-NC-SA 4.0 (the card notes original source datasets may have separate terms).
- Published total: 21,139 aligned triples; train 16,911 and test 4,228.
- Fields: `id`, `english`, `hindi`, `hinglish`.
- Intended derived labels: `english` -> likely LATIN, `hindi` -> likely DEVANAGARI, and `hinglish` -> derive from actual letter counts. These are expectations to validate programmatically, not assumed ground truth.
- The dataset card notes that much of the Hindi side is machine-translated and Romanized Hinglish spelling is not standardized.
- License warning: non-commercial and share-alike terms can constrain downstream use. Review all original-source terms too.
- Citation: Mazumder et al., “Neither Here Nor There: Cross-Lingual Representation Dynamics of Code-Mixed Text in Multilingual Encoders,” arXiv:2603.19771 / EMNLP Findings 2026.
- Source: https://huggingface.co/datasets/debajyotimaz/trilingual-hinglish-corpus

## 3. Critical modeling gate

The current checkpoint `google/bert_uncased_L-2-H-128-A-2` is English-oriented. Its tokenizer coverage for Devanagari has not yet been measured in the user's runtime. Run:

```bash
python -m src.data.audit_tokenizer
```

If Devanagari/mixed examples are mostly mapped to unknown tokens, do not fine-tune this checkpoint. Consider a script-capable multilingual encoder, or benchmark a character n-gram TF-IDF + Linear SVM baseline, which is lightweight and directly represents writing patterns.

## 4. Held-out evaluation policy

The user's `call_data(3).txt` remains reserved for final real-world evaluation only. Do not use it for training, validation, model selection, threshold tuning, or label-rule tuning. Do not upload it to this repository. Other scripts (Gurmukhi, Telugu, Malayalam, etc.) should be marked OUT_OF_SCOPE unless a separate explicit label is added.
