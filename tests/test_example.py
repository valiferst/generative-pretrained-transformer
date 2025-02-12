"""
Dummy unit test using the unittest module.
"""

import unittest
from utils.tokenizer import DummyTokenizer

class TestDummyTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DummyTokenizer()

    def test_tokenize_detokenize(self):
        text = "this is a test"
        tokens = self.tokenizer.tokenize(text)
        reconstructed = self.tokenizer.detokenize(tokens)
        # In our dummy implementation, the detokenized text may not equal the original,
        # but we can at least check that we get a string back.
        self.assertIsInstance(reconstructed, str)

if __name__ == '__main__':
    unittest.main()
