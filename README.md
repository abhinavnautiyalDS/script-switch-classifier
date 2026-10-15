# Script Switch Classifier

Lightweight classifier for deciding whether a text utterance is predominantly written in Devanagari or Latin script. Mixed-script utterances are assigned according to the dominant alphabetic-character count.

## Important project rules

- Keep `call_data(3).txt` strictly held out for final real-world evaluation. Do not use it for training, validation, or model selection.
- Audit tokenizer coverage before fine-tuning. The initial BERT-Tiny checkpoint is English-oriented and may not support Devanagari adequately.
- Inspect public dataset schemas, licenses, and splits before using them.
- Detect unsupported scripts separately where feasible rather than silently treating them as Latin.

## First steps

1. Inspect dataset schemas and available splits.
2. Audit tokenizer coverage on Devanagari, Latin, and mixed examples.
3. Train only after the tokenizer passes the coverage gate.
4. Evaluate on the held-out call transcripts only after model selection is complete.

No accuracy claims should be made until measured on an appropriate evaluation set.
