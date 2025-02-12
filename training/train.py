"""
Dummy training script for the GPT model.
"""

import torch
from model.transformer import DummyTransformer
from training.config import config
from utils.data_loader import load_dummy_data

def train():
    # Load dummy data
    data = load_dummy_data()

    # Initialize the model
    model = DummyTransformer(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"]
    )

    # Dummy optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()

    # Dummy training loop
    model.train()
    for epoch in range(config["epochs"]):
        optimizer.zero_grad()
        outputs = model(data["input"])
        loss = loss_fn(outputs.view(-1, config["vocab_size"]), data["target"].view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
