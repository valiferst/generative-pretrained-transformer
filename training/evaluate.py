"""
Dummy evaluation script for the GPT model.
"""

import torch
from model.transformer import DummyTransformer
from training.config import config
from utils.data_loader import load_dummy_data

def evaluate():
    # Load dummy data
    data = load_dummy_data()

    # Initialize the model
    model = DummyTransformer(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"]
    )

    # Dummy evaluation (no real metrics calculated)
    model.eval()
    with torch.no_grad():
        outputs = model(data["input"])
    print("Evaluation complete. Dummy outputs obtained.")

if __name__ == "__main__":
    evaluate()
