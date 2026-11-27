"""Load real MNIST as numpy arrays for the Day 3 video (no torchvision needed).

Uses Michael Nielsen's public mirror (50k train + 10k validation + 10k test).
Train + validation are merged, which gives the standard 60,000 / 10,000 split.
The file is cached in ./data (git-ignored) after the first download.
"""
import gzip
import os
import pickle
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist.npz")


def load_mnist():
    """Returns x_train (60000,28,28) uint8, y_train, x_test (10000,28,28) uint8, y_test."""
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        raw = CACHE + ".pkl.gz"
        urllib.request.urlretrieve(URL, raw)
        with gzip.open(raw, "rb") as f:
            tr, va, te = pickle.load(f, encoding="latin1")
        to_img = lambda a: (a * 255).round().astype("uint8").reshape(-1, 28, 28)
        np.savez_compressed(
            CACHE,
            x_train=to_img(np.concatenate([tr[0], va[0]])), y_train=np.concatenate([tr[1], va[1]]).astype("uint8"),
            x_test=to_img(te[0]), y_test=te[1].astype("uint8"),
        )
        os.remove(raw)
    d = np.load(CACHE)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


if __name__ == "__main__":
    xt, yt, xs, ys = load_mnist()
    print(f"train {xt.shape} {xt.dtype}, test {xs.shape}, first test label = {ys[0]}")
