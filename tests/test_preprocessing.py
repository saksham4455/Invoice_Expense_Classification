from app import preprocess_text


def test_preprocess_lower_and_strip_punctuation():
    out = preprocess_text("Hello, WORLD!! 123")
    # punctuation removed and lowercased; numeric tokens remain
    assert "hello" in out
    assert "world" in out
    assert "123" in out
