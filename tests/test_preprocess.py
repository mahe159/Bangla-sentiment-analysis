from bangla_sentiment import normalize


def test_non_string_becomes_empty():
    assert normalize(None) == ""
    assert normalize(float("nan")) == ""


def test_urls_and_invisible_characters_removed():
    assert normalize("দেখুন https://example.com/x ভালো\u200c") == "দেখুন ভালো"


def test_bangla_digits_become_ascii():
    assert normalize("দাম ৫০০ টাকা") == "দাম 500 টাকা"


def test_repeated_punctuation_collapsed_and_spaces_squeezed():
    assert normalize("দারুণ!!!   বই।।।") == "দারুণ! বই।"


def test_latin_text_lowercased():
    assert normalize("Very GOOD") == "very good"
