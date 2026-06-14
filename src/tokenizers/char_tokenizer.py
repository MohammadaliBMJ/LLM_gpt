class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, texts: list[str]):
        """train the character tokenizer on the given texts

        Args:
            texts (list[str]): list of texts to train the tokenizer on
        """
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(chars)
        chars = ['<PAD>', '<UNK>'] + chars  # Add special tokens for padding and unknown characters
        self.char_to_id = {char: idx for idx, char in enumerate(chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

    def encode(self, text: str):
        """encode input text using character tokenizer

        Args:
            text (str): input text to encode

        Returns:
            list[int]: list of token IDs representing the encoded text.
        """
        return [self.char_to_id.get(char, self.char_to_id['<UNK>']) for char in text]

    def decode(self, ids: list[int]):
        """decode the input ids to characters

        Args:
            ids (list[int]): list of token IDs to decode.

        Returns:
            str: the decoded text.
        """
        return ''.join([self.id_to_char[id] for id in ids if id in self.id_to_char])