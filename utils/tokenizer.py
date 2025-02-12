"""
Dummy tokenizer implementation.
"""

class DummyTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            # Create a dummy vocabulary
            self.vocab = {f"word{i}": i for i in range(10000)}
        else:
            self.vocab = vocab

    def tokenize(self, text):
        # Dummy tokenization: split by whitespace and convert to token IDs
        tokens = text.split()
        return [self.vocab.get(token, 0) for token in tokens]

    def detokenize(self, token_ids):
        # Dummy detokenization: convert token IDs back to string (reverse lookup)
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return " ".join([reverse_vocab.get(i, "<UNK>") for i in token_ids])

if __name__ == "__main__":
    tokenizer = DummyTokenizer()
    sample_text = "this is a dummy example"
    token_ids = tokenizer.tokenize(sample_text)
    print("Token IDs:", token_ids)
    print("Detokenized:", tokenizer.detokenize(token_ids))
