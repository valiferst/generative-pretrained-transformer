"""
Dummy data loader.
"""

import torch

def load_dummy_data():
    # Create dummy input and target tensors
    # For example: Batch size 2, sequence length 10
    dummy_input = torch.randint(0, 10000, (2, 10))
    dummy_target = torch.randint(0, 10000, (2, 10))
    return {"input": dummy_input, "target": dummy_target}

if __name__ == "__main__":
    data = load_dummy_data()
    print("Dummy data loaded:")
    print("Input:", data["input"])
    print("Target:", data["target"])

