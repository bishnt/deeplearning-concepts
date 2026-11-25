"""Day 1: the Perceptron, implemented from scratch in PyTorch.

A perceptron computes  y = step(w . x + b)  and learns with Rosenblatt's rule:
    w <- w + lr * (target - prediction) * x
    b <- b + lr * (target - prediction)
It can only draw one straight line, so it solves AND / OR but not XOR.
"""
import torch


class Perceptron:
    def __init__(self, n_inputs: int, lr: float = 0.1):
        self.w = torch.zeros(n_inputs)
        self.b = torch.zeros(1)
        self.lr = lr

    @staticmethod
    def step(z: torch.Tensor) -> torch.Tensor:
        """Step activation: 1 if z > 0 else 0."""
        return (z > 0).float()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(x @ self.w + self.b)

    def fit(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 100) -> int:
        """Train with the perceptron rule. Returns the epoch it converged at,
        or -1 if it never reached zero errors."""
        for epoch in range(1, epochs + 1):
            errors = 0
            for xi, yi in zip(x, y):
                error = yi - self.predict(xi)
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
            if errors == 0:
                return epoch
        return -1


GATES = {
    "AND": torch.tensor([0.0, 0.0, 0.0, 1.0]),
    "OR": torch.tensor([0.0, 1.0, 1.0, 1.0]),
    "XOR": torch.tensor([0.0, 1.0, 1.0, 0.0]),
}
INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])


def main():
    for name, y in GATES.items():
        p = Perceptron(n_inputs=2)
        converged = p.fit(INPUTS, y, epochs=100)
        preds = p.predict(INPUTS)
        acc = (preds == y).float().mean().item()
        status = f"converged at epoch {converged}" if converged > 0 else "did NOT converge"
        print(f"{name:>3}: w={p.w.tolist()} b={p.b.item():.2f} acc={acc:.0%} ({status})")


if __name__ == "__main__":
    main()
