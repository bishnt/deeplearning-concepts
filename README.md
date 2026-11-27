# Deep Learning Concepts

From-scratch PyTorch implementations for the series *Explaining Deep Learning Concepts*.

| Day | Concept | Code |
|-----|---------|------|
| 1 | Perceptron | [`01_perceptron/perceptron.py`](01_perceptron/perceptron.py) |
| 2 | MLP (solves XOR) | [`02_mlp/mlp.py`](02_mlp/mlp.py) |
| 3 | Neural network for MNIST digits (architecture + loss) | [`03_neural_network/digit_net.py`](03_neural_network/digit_net.py) |

Run: `pip install -r requirements.txt && python 01_perceptron/perceptron.py`

Day 2: `python 02_mlp/mlp.py`

Day 3: `python 03_neural_network/digit_net.py`

### Day 3 video (Manim)

Rendered video (1080p30, 2:57, no audio): [`03_neural_network/video/out/Day3_full.mp4`](03_neural_network/video/out/Day3_full.mp4),
plus one clip per script section in the same folder (`S1_Hook` ... `S6_Wrap`, lengths match the script timestamps).

Re-render: `pip install manim && python 03_neural_network/video/render_all.py`
(needs ffmpeg and the Pango/Cairo system libraries; real MNIST digits are fetched by `video/get_mnist.py`).
