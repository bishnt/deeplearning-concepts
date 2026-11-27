"""Day 2: the Multi-Layer Perceptron (MLP), implemented from scratch in PyTorch.

Day 1's perceptron draws a single straight line, so it can't solve XOR.
Stacking a hidden layer with a non-linear activation fixes that:
    h = sigmoid(x @ W1 + b1)      hidden layer
    y = sigmoid(h @ W2 + b2)      output layer
Training uses backpropagation, written out by hand (no autograd, no nn.Module)
so every gradient is visible:
    dL/dz2 = (y - t) * y * (1 - y)             (MSE loss, sigmoid output)
    dL/dz1 = (dL/dz2 @ W2.T) * h * (1 - h)     (chain rule through hidden layer)
"""
import torch


class MLP:
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int = 1, lr: float = 2.0):
        self.W1 = torch.randn(n_inputs, n_hidden)
        self.b1 = torch.zeros(n_hidden)
        self.W2 = torch.randn(n_hidden, n_outputs)
        self.b2 = torch.zeros(n_outputs)
        self.lr = lr

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-z))

    def forward(self, x: torch.Tensor):
        """Returns (hidden activations, output). Hidden is kept for backprop."""
        h = self.sigmoid(x @ self.W1 + self.b1)
        y = self.sigmoid(h @ self.W2 + self.b2)
        return h, y

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return (self.forward(x)[1] > 0.5).float()

    def fit(self, x: torch.Tensor, t: torch.Tensor, epochs: int = 5000, log_every: int = 1000):
        """Full-batch gradient descent with hand-written backprop."""
        n = x.shape[0]
        for epoch in range(1, epochs + 1):
            h, y = self.forward(x)
            loss = ((y - t) ** 2).mean()

            # backward pass
            dz2 = (y - t) * y * (1 - y)          # (n, n_outputs)
            dW2 = h.T @ dz2 / n
            db2 = dz2.mean(dim=0)
            dz1 = (dz2 @ self.W2.T) * h * (1 - h)  # (n, n_hidden)
            dW1 = x.T @ dz1 / n
            db1 = dz1.mean(dim=0)

            # gradient descent step
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if epoch % log_every == 0 or epoch == 1:
                print(f"epoch {epoch:>5}  loss={loss.item():.4f}")


INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


def main():
    torch.manual_seed(42)
    model = MLP(n_inputs=2, n_hidden=4)
    model.fit(INPUTS, XOR, epochs=5000)

    _, probs = model.forward(INPUTS)
    preds = model.predict(INPUTS)
    acc = (preds == XOR).float().mean().item()

    print("\nXOR results:")
    for xi, p, yi in zip(INPUTS, probs, XOR):
        print(f"  {xi.tolist()} -> {p.item():.3f}  (target {int(yi.item())})")
    print(f"accuracy = {acc:.0%}  (the single perceptron from Day 1 maxes out at 75%)")


if __name__ == "__main__":
    main()
