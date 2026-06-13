class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text):
        """Encode text to digits.

        Args:
            text (str): input text to tokenize.

        Returns:
            list: list of integers representing the input text.
        """
        return list(text.encode('utf-8'))

    def decode(self, tokens):
        """Decode tokens to text.

        Args:
            tokens (list): list of integers representing the input text.

        Returns:
            str: decoded text.
        """
        return bytes(tokens).decode('utf-8')