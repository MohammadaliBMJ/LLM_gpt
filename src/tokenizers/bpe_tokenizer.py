class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

        self.next_token_id = 256  # Start assigning token IDs from 256
        self.vocabs = {i for i in range(256)}  # Initialize vocab with byte tokens
        self.merges = []    

    def train(self, texts: str):
        """Train the BPE tokenizer on the texts."""
        text_bytes = [list(text.encode('utf-8')) for text in texts]

        while len(self.vocabs) < self.vocab_size:
            pair_freq = {}
            for byte_seq in text_bytes:
                for i in range(len(byte_seq) - 1):
                    pair = (byte_seq[i], byte_seq[i + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + 1

            if not pair_freq:
                break

            most_freq_pair = max(pair_freq, key=pair_freq.get)
            self.merges.append((most_freq_pair[0], most_freq_pair[1], self.next_token_id))

            new_token_id = self.next_token_id
            self.next_token_id += 1
            self.vocabs.add(new_token_id)

            for byte_seq in text_bytes:
                i = 0
                while i < len(byte_seq) - 1:
                    if (byte_seq[i], byte_seq[i + 1]) == most_freq_pair:
                        byte_seq[i] = new_token_id
                        del byte_seq[i + 1]
                    else:
                        i += 1

    def encode(self, text: str):
        """Encode the text into a list of token IDs."""
        text_bytes = list(text.encode('utf-8'))
        for pair_1, pair_2, new_token_id in self.merges:
            i = 0
            while i < len(text_bytes) - 1:
                if (text_bytes[i], text_bytes[i + 1]) == (pair_1, pair_2):
                    text_bytes[i] = new_token_id
                    del text_bytes[i + 1]
                else:
                    i += 1
        return text_bytes

    def decode(self, tokens):
        """Decode a list of token IDs back into a string."""
        for pair_1, pair_2, new_token_id in reversed(self.merges):
            i = 0
            while i < len(tokens):
                if tokens[i] == new_token_id:
                    tokens[i] = pair_1
                    tokens.insert(i + 1, pair_2)
                    i += 2
                else:
                    i += 1
        return bytes(tokens).decode('utf-8')