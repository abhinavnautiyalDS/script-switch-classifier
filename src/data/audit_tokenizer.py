"""Audit tokenizer behavior before training; this script does not train a model."""
from transformers import AutoTokenizer

MODEL_NAME = "google/bert_uncased_L-2-H-128-A-2"
EXAMPLES = {
    "devanagari": "नमस्ते सर, आप कैसे हैं?",
    "latin_romanized_hindi": "Namaste sir, aap kaise hain?",
    "mixed": "नमस्ते sir, aap कैसे हैं?",
    "english": "Hello sir, how are you?",
}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    unk_id = tokenizer.unk_token_id
    print(f"Model: {MODEL_NAME}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}; UNK={tokenizer.unk_token!r}")

    total_unk = total_tokens = 0
    for name, text in EXAMPLES.items():
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        unknown = sum(token_id == unk_id for token_id in ids)
        print(f"\n[{name}] {text}")
        print(f"  tokens: {tokens}")
        print(f"  unknown tokens: {unknown}/{len(ids)}")
        if name in {"devanagari", "mixed"}:
            total_unk += unknown
            total_tokens += len(ids)

    unk_rate = total_unk / max(total_tokens, 1)
    print(f"\nDevanagari/mixed unknown-token rate: {unk_rate:.1%}")
    if unk_rate > 0.05:
        print("GATE: FAIL — do not fine-tune this checkpoint; select a script-capable tokenizer.")
    else:
        print("GATE: REVIEW — inspect tokenization above before proceeding.")
    print("Note: token-level UNK rate is not the same as character coverage.")


if __name__ == "__main__":
    main()
