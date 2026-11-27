"""Day 3: a neural network that reads handwritten digits (MNIST), in PyTorch.

Days 1 and 2 were written from scratch. From today PyTorch's own components
(torch.nn) do the bookkeeping:
    nn.Flatten            28 x 28 image  ->  784 numbers
    nn.Linear             fully connected layer,  z = x @ W.T + b
    nn.ReLU               max(0, z), the non-linearity between layers
    nn.CrossEntropyLoss   softmax + (-log of the correct digit's probability)

Architecture:   784 -> 128 -> ReLU -> 64 -> ReLU -> 10
    784*128 + 128  = 100,480
    128*64  + 64   =   8,256
    64*10   + 10   =     650
                     -------
                     109,386 parameters, all random at the start.

This video covers the model and the loss. Optimization, the training loop and
evaluation come in the next video.
"""
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH_SIZE = 64


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),          # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(784, 128),   # 784 is fixed by the data, 128 is our choice
            nn.ReLU(),             # without it, all layers collapse into one linear function
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),     # one raw score (logit) per digit 0-9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch records every step here so backpropagation can replay it later
        return self.layers(x)


def get_loaders(batch_size: int = BATCH_SIZE):
    """MNIST as tensors. Pixels go from 0-255 to 0-1 (ToTensor), which keeps training stable."""
    tf = transforms.ToTensor()
    train = datasets.MNIST("data", train=True, download=True, transform=tf)
    test = datasets.MNIST("data", train=False, download=True, transform=tf)
    # train gets shuffled; test stays untouched, the model must never see it while learning
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
    )


def loss_demo():
    """CrossEntropyLoss = softmax, then -log(probability of the correct digit)."""
    print("\nCross entropy: confident and wrong is punished hard")
    for p in (0.9, 0.01):
        print(f"  p(correct) = {p:<4}  ->  loss = -log({p}) = {-math.log(p):.2f}")

    logits = torch.tensor([[2.0, 1.0, 0.1]])          # raw scores for 3 classes
    probs = torch.softmax(logits, dim=1)              # exp(z) / sum(exp(z))
    target = torch.tensor([0])
    by_hand = -torch.log(probs[0, target])
    built_in = nn.CrossEntropyLoss()(logits, target)  # takes logits, so no softmax in the model
    print(f"  softmax -> {[round(v, 3) for v in probs[0].tolist()]}  (sums to {probs.sum().item():.0f})")
    print(f"  by hand = {by_hand.item():.4f}   CrossEntropyLoss = {built_in.item():.4f}")


def main():
    torch.manual_seed(42)  # same random start every run -> reproducible results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPUs excel at matrix multiplication
    print(f"device: {device}")

    train_loader, test_loader = get_loaders()
    print(f"train: {len(train_loader.dataset):,} images -> {len(train_loader)} batches of {BATCH_SIZE}")
    print(f"test : {len(test_loader.dataset):,} images (kept aside for evaluation)")

    model = DigitNet().to(device)
    print(f"\n{model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, labels)
    print(f"\none batch: images {tuple(images.shape)} -> logits {tuple(logits.shape)}")
    print(f"untrained loss = {loss.item():.3f}  (random guessing over 10 digits = ln(10) = {math.log(10):.3f})")

    loss_demo()


if __name__ == "__main__":
    main()
