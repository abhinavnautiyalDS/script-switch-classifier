from src.data.script_labels import count_scripts, label_text


def test_pure_devanagari():
    assert label_text("नमस्ते सर, आप कैसे हैं?") == "DEVANAGARI"


def test_latin_romanized_hindi():
    assert label_text("Namaste sir, aap kaise hain?") == "LATIN"


def test_mixed_devanagari_dominant():
    assert label_text("नमस्ते sir, आप कैसे हैं?") == "DEVANAGARI"


def test_mixed_latin_dominant():
    assert label_text("Hello जी, how are आप?") == "LATIN"


def test_tie_is_explicit():
    assert label_text("Hi जी") == "TIE"


def test_devanagari_vowel_marks_are_counted():
    result = count_scripts("कि")
    assert result.devanagari == 2
    assert result.label == "DEVANAGARI"


def test_numbers_and_punctuation_do_not_count():
    result = count_scripts("123!!! नमस्ते")
    assert result.devanagari > 0
    assert result.latin == 0
    assert result.label == "DEVANAGARI"


def test_other_scripts_not_mislabelled_latin():
    assert label_text("ਸਤ ਸ੍ਰੀ ਅਕਾਲ") == "OUT_OF_SCOPE"


def test_empty_input():
    assert label_text("... 123 ") == "NO_LETTERS"
