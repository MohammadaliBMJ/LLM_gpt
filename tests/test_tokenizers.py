import pytest
from src.tokenizers.byte_tokenizer import ByteTokenizer
from src.tokenizers.bpe_tokenizer import BPETokenizer


# Byte tokenizer tests
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

# BPE tokenizer tests
@pytest.fixture
def bpe_tokenizer():
    """Return tokens using BPETokenizer.

    Returns:
        BPETokenizer: An instance of the BPETokenizer.
    """
    return BPETokenizer(vocab_size=266)

def test_bpe_tokenizer_vocabs(bpe_tokenizer):
    """Test the BPETokenizer initializes with the correct vocabs and next token ID."""
    assert len(bpe_tokenizer.vocabs) == 256, f"Expected initial vocab size 256, but got {len(bpe_tokenizer.vocabs)}"
    assert bpe_tokenizer.next_token_id == 256, f"Expected next token ID 256, but got {bpe_tokenizer.next_token_id}"

def test_bpe_tokenizer_train(bpe_tokenizer):
    """Test the BPETokenizer can train on a simple text."""
    texts = ["hello this is a test for BPE tokenizer", "another test sentence for our tokenizer"]
    bpe_tokenizer.train(texts)
    assert len(bpe_tokenizer.merges) > 0, "Expected merges to be added after training"
    assert len(bpe_tokenizer.vocabs) > 256, "Expected vocab size to increase after training"

def test_bpe_tokenizer_encode(bpe_tokenizer):
    """Test the BPETokenizer can encode text after training."""
    texts = ["hello this is a test for BPE tokenizer", "another test sentence for our tokenizer"]
    bpe_tokenizer.train(texts)
    tokens = bpe_tokenizer.encode("hello this is a test")
    assert isinstance(tokens, list), f"Expected tokens to be a list, got {type(tokens)}"
    assert all(isinstance(token, int) for token in tokens), "Expected all tokens to be integers"

def test_bpe_tokenizer_decode(bpe_tokenizer):
    """Test the BPETokenizer can decode tokens back to text."""
    texts = ["hello this is a test for BPE tokenizer", "another test sentence for our tokenizer"]
    bpe_tokenizer.train(texts)
    tokens = bpe_tokenizer.encode("hello this is a test")
    decoded_text = bpe_tokenizer.decode(tokens)
    assert isinstance(decoded_text, str), f"Expected decoded text to be a string, got {type(decoded_text)}"
    assert decoded_text == "hello this is a test", f"Expected 'hello this is a test', but got '{decoded_text}'"
