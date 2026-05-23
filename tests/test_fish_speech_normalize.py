import pytest
from atom.models.fish_speech.prompt_utils import normalize_fish_speech_text, _python_normalize_fish_speech_text

def test_normalization_parity():
    test_cases = [
        ("Hello world", False, "Hello world"),
        ("Hello world", True, "<|speaker:0|>Hello world"),
        ("Hello <speaker:2> world", False, "Hello <|speaker:2|> world"),
        ("Hello <speaker:2> world", True, "Hello <|speaker:2|> world"),
        ("Hello <|speaker:3|> world", False, "Hello <|speaker:3|> world"),
        ("Hello <|speaker:3|> world", True, "Hello <|speaker:3|> world"),
        ("<speaker:4><speaker:5>", False, "<|speaker:4|><|speaker:5|>"),
    ]

    for text, add_default, expected in test_cases:
        rust_out = normalize_fish_speech_text(text, add_default_speaker=add_default)
        python_out = _python_normalize_fish_speech_text(text, add_default_speaker=add_default)
        assert rust_out == python_out, f"Mismatch for '{text}': rust={rust_out}, python={python_out}"
        assert rust_out == expected, f"Expected {expected}, got {rust_out}"

def test_normalization_invalid():
    invalid_texts = [
        "Hello <|invalid|> world",
        "Hello <|speaker:2|> and <|other_token|>",
    ]

    for text in invalid_texts:
        with pytest.raises(ValueError):
            normalize_fish_speech_text(text)
        with pytest.raises(ValueError):
            _python_normalize_fish_speech_text(text)
