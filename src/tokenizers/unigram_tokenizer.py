import math

class UnigramTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab_count = {}

    def train(self, texts: list[str]):
        """Train unigram on texts to build the vocabularies and calculate token probabilities.

        Args:
            texts (list[str]): List of input texts to train the tokenizer on.
        """
        for text in texts:
            for word in text.split():
                for i in range(len(word) + 1):
                    for j in range(i + 1, len(word) + 1):
                        substring = word[i:j]
                        self.vocab_count[substring] = self.vocab_count.get(substring, 0) + 1

        sorted_vocab_count = sorted(self.vocab_count.items(), key=lambda item: item[1], reverse=True)
        self.vocab = [word for word, _ in sorted_vocab_count[:self.vocab_size]]

        # Probabilities
        tokens_total_count = sum(self.vocab_count[token] for token in self.vocab)
        self.token_probs = {token: self.vocab_count[token] / tokens_total_count for token in self.vocab}

    def segment(self, word: str) -> list[str]:
        """Segment each word into the best splits with highest probability.

        Args:
            word (str): The input word to be segmented.

        Returns:
            list[str]: List of segmented tokens for the input word.
        """
        word_length = len(word)
        dp = [float('inf')] * (word_length + 1)
        dp[0] = 0
        backtrack = [-1] * (word_length + 1)
        for i in range(1, word_length + 1):
            for j in range(i):
                substring = word[j:i]
                if substring in self.token_probs:
                    prob = self.token_probs[substring]
                    cost = dp[j] - math.log(prob)
                    if cost < dp[i]:
                        dp[i] = cost
                        backtrack[i] = j

        if backtrack[word_length] == -1:
            return list(word)

        segments = []
        idx = word_length
        while idx > 0:
            start = backtrack[idx]
            segments.append(word[start:idx])
            idx = start

        segments.reverse()
        return segments
    
    def encode(self, text: str) -> list[str]:
        """Encode the text to tokens using segment(). 

        Args:
            text (str): The input text to be encoded.

        Returns:
            list[str]: List of tokens representing the encoded text.
        """
        tokens = []
        for word in text.split():
            tokens.extend(self.segment(word))
            tokens.append('<SPACE>')  # Add space as a token between words
        return tokens

    def decode(self, tokens: list[str]) -> str:
        """Decode the tokens back to text.

        Args:
            tokens (list[str]): List of tokens to be decoded.

        Returns:
            str: The decoded text.
        """
        text = ''
        for token in tokens:
            if token == '<SPACE>':
                text += ' '
            else:
                text += token
        return text.strip()