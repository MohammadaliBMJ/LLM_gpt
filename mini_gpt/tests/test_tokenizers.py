import pytest
from src.tokenizers.byte_tokenizer import ByteTokenizer


# Bye tokenizer tests
@pytest.fixture
def byte_tokenizer():
    """Return tokens using ByteTokenizer.

    Returns:
        ByteTokenizer: An instance of the ByteTokenizer.
    """
    return ByteTokenizer()

def test_byte_tokenizer_encode_int(byte_tokenizer):
    """Test the ByteTokenizer encoder returns a list of integers."""
    text = "Transformers are amazing!"
    tokens = byte_tokenizer.encode(text)
    for token in tokens:
        assert isinstance(token, int), f"Expected token to be an integer, got {type(token)}"

def test_byte_tokenizer_encode_range(byte_tokenizer):
    """Test the ByteTokenizer encoder returns tokens in the range [0, 255]."""
    text = "Learning LLMs is fun!"
    tokens = byte_tokenizer.encode(text)
    assert len(tokens) > 0, "Expected non-empty token list"
    for token in tokens:
        assert 0 <= token <= 255, f"Expected token to be in range [0, 255], got {token}"

def test_byte_tokenizer_decode_string(byte_tokenizer):
    """Test the ByteTokenizer decoder returns a string."""
    text = "Transformers are amazing!"
    tokens = byte_tokenizer.encode(text)
    decoded_text = byte_tokenizer.decode(tokens)
    assert type(decoded_text) == str, f"Expected decoded text to be a string, got {type(decoded_text)}"

def test_byte_tokenizer_round_trip(byte_tokenizer):
    """Test the ByteTokenizer can encode and decode correctly."""
    text = "Transformers are amazing!"
    tokens = byte_tokenizer.encode(text)
    decoded_text = byte_tokenizer.decode(tokens)
    assert text == decoded_text, f"Expected '{text}', but got '{decoded_text}'"

def test_byte_tokenizer_empty_string(byte_tokenizer):
    """Test the ByteTokenizer can handle an empty string."""
    text = ""
    tokens = byte_tokenizer.encode(text)
    decoded_text = byte_tokenizer.decode(tokens)
    assert text == decoded_text, f"Expected empty string, but got '{decoded_text}'"

def test_byte_tokenizer_vocab_size(byte_tokenizer):
    """Test the ByteTokenizer has the correct vocabulary size."""
    assert byte_tokenizer.vocab_size == 256, f"Expected vocabulary size 256, but got {byte_tokenizer.vocab_size}"