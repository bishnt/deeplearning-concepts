"""Day 3 video (Explaining Deep Learning Concepts): a neural network for MNIST digits.

One Manim scene per script section, each timed to the script's timestamps:
    S1_Hook      0:00 - 0:08   (8 s)
    S2_Build     0:08 - 0:56   (48 s)   what we're building
    S3_Data      0:56 - 1:36   (40 s)   setup and data
    S4_Model     1:36 - 2:11   (35 s)   the model
    S5_Loss      2:11 - 2:47   (36 s)   loss function
    S6_Wrap      2:47 - 2:57   (10 s)   wrap-up
Real MNIST digits are used everywhere (see get_mnist.py). Text only, no LaTeX needed.
Render everything with:  python render_all.py
"""
import os
import sys

import numpy as np
from manim import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_mnist import load_mnist  # noqa: E402

BG = "#101418"
config.background_color = BG
BLUE, GREEN, YELLOW, RED, GRAY_C, PURPLE_C, ORANGE_C = (
    "#58C4DD", "#83C167", "#FFD866", "#FC6255", "#9AA4AD", "#B18CF0", "#FF9F43")
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"

XTR, YTR, XTE, YTE = load_mnist()
SEVEN = XTE[0]                                   # first MNIST test image is a 7
RNG = np.random.default_rng(3)


# ----------------------------------------------------------------------------- helpers
def T(s, size=30, color=WHITE, **kw):
    return Text(s, font=SANS, font_size=size, color=color, **kw)


def M(s, size=24, color=WHITE, **kw):
    return Text(s, font=MONO, font_size=size, color=color, **kw)


def img(arr, width):
    """Pixel-sharp ImageMobject from a 2D uint8 array."""
    a = np.asarray(arr, dtype=np.uint8)
    a = np.stack([a] * 3, axis=-1)
    m = ImageMobject(a)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.scale_to_fit_width(width)
    return m


def pixel_grid(m, n=28, color=GRAY_C):
    left, top, size = m.get_left()[0], m.get_top()[1], m.width
    g = VGroup()
    for i in range(n + 1):
        t = i * size / n
        g.add(Line([left + t, top, 0], [left + t, top - size, 0], stroke_width=0.5, color=color, stroke_opacity=0.6))
        g.add(Line([left, top - t, 0], [left + size, top - t, 0], stroke_width=0.5, color=color, stroke_opacity=0.6))
    return g


def pill(s, color, size=20):
    t = T(s, size, color)
    r = RoundedRectangle(corner_radius=0.15, width=t.width + 0.45, height=t.height + 0.28,
                         stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=0.12)
    t.move_to(r)
    return VGroup(r, t)


def code_rows(lines, size=22, t2c=None, buff=0.15):
    """Monospace code as one Text per line, indentation preserved, left aligned."""
    cw = M("M" * 20, size).width / 20
    rows = VGroup()
    for ln in lines:
        ind = len(ln) - len(ln.lstrip(" "))
        t = M(ln.strip(), size, t2c=t2c or {})
        t.indent = ind
        rows.add(t)
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    for r in rows:
        r.shift(RIGHT * r.indent * cw)
    return rows


KEYWORDS = {"class": PURPLE_C, "def": PURPLE_C, "return": PURPLE_C, "import": PURPLE_C, "True": ORANGE_C,
            "False": ORANGE_C, "True,": ORANGE_C}


def code_panel(lines, center, size=22, t2c=None, pad=0.3):
    rows = code_rows(lines, size, t2c)
    rows.move_to(center)
    box = RoundedRectangle(corner_radius=0.15, width=rows.width + 2 * pad, height=rows.height + 2 * pad,
                           stroke_color=GRAY_C, stroke_width=1.5, fill_color="#171C22", fill_opacity=1)
    box.move_to(rows)
    return box, rows


class Day3Scene(Scene):
    """Every scene pads itself to its script timestamp, then fades out in the last 0.4 s."""
    TARGET = 10.0

    def construct(self):
        self.add(T("Day 3/100", 22, GRAY_C).to_corner(UR, buff=0.35))
        self.build()
        used = self.renderer.time
        print(f"[{type(self).__name__}] content ends at {used:.2f}s (target {self.TARGET}s)")
        left = self.TARGET - 0.4 - used
        if left > 0:
            self.wait(left)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.4)


# ----------------------------------------------------------------------------- the network drawing
class Net:
    """784 -> 128 -> 64 -> 10 drawn as columns of neurons (visible subset + dots)."""
    XS = [-3.4, -0.9, 1.6, 4.1]

    def __init__(self, cy=0.25):
        ys = [[1.65, 1.1, 0.55, -0.55, -1.1, -1.65], [1.65, 1.1, 0.55, -0.55, -1.1, -1.65],
              [1.1, 0.55, -0.55, -1.1], [2.475 - 0.55 * i for i in range(10)]]
        info = [("784", "Input", BLUE), ("128", "Hidden 1", YELLOW), ("64", "Hidden 2", YELLOW), ("10", "Output", BLUE)]
        self.pos, self.nodes, self.dots, self.labels, self.numlabels = [], [], VGroup(), [], []
        for l, (x, yl) in enumerate(zip(self.XS, ys)):
            col = info[l][2]
            r = 0.22 if l == 3 else 0.17
            g = VGroup()
            pts = []
            for k, y in enumerate(yl):
                c = Circle(radius=r, stroke_color=col, stroke_width=2, fill_color=col, fill_opacity=0.18)
                c.move_to([x, y + cy, 0])
                g.add(c)
                pts.append(c.get_center())
                if l == 3:
                    g.add(T(str(k), 18, WHITE).move_to(c))
            if l < 3:
                self.dots.add(T("⋮", 32, GRAY_C).move_to([x, cy, 0]))
            self.pos.append(pts)
            self.nodes.append(g)
            num = T(info[l][0], 32, col)
            lab = VGroup(num, T(info[l][1], 20, GRAY_C)).arrange(DOWN, buff=0.06).move_to([x, cy - 3.05, 0])
            self.labels.append(lab)
            self.numlabels.append(num)
        self.edges, self.edge_map = [], []
        for l in range(3):
            rr = 0.22 if l + 1 == 3 else 0.17
            grp, mp = VGroup(), []
            for i, a in enumerate(self.pos[l]):
                for j, b in enumerate(self.pos[l + 1]):
                    ln = Line(a + RIGHT * 0.17, b + LEFT * rr, stroke_width=0.8, color=GRAY_C, stroke_opacity=0.35)
                    grp.add(ln)
                    mp.append((i, j, ln))
            self.edges.append(grp)
            self.edge_map.append(mp)
        mids = [(self.XS[i] + self.XS[i + 1]) / 2 for i in range(3)]
        self.pills = [pill("Linear + ReLU", ORANGE_C).move_to([mids[0], cy + 3.4, 0]),
                      pill("Linear + ReLU", ORANGE_C).move_to([mids[1], cy + 3.4, 0]),
                      pill("Linear", BLUE).move_to([mids[2], cy + 3.4, 0])]
        self.cy = cy

    def body(self):
        """Everything that scales/moves together (not the pills/labels)."""
        return VGroup(*self.edges, *self.nodes, self.dots, *self.labels, *self.pills)


# ============================================================================= S1  Hook  (0:00 - 0:08)
class S1_Hook(Day3Scene):
    TARGET = 8.0

    def build(self):
        idx = RNG.choice(60000, 55, replace=False)
        tiles = Group(*[img(XTR[i], 0.95) for i in idx])
        tiles.arrange_in_grid(5, 11, buff=0.1).set_opacity(0.55)
        self.play(LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=0.02), run_time=1.4)

        backdrop = RoundedRectangle(corner_radius=0.3, width=8.6, height=3.2, stroke_width=0,
                                    fill_color=BG, fill_opacity=0.88)
        counter = ValueTracker(0)
        num = always_redraw(lambda: T(f"{int(counter.get_value()):,}", 96, YELLOW, weight=BOLD).move_to([0, 0.35, 0]))
        sub = T("handwritten digit images", 36, WHITE).move_to([0, -0.9, 0])
        self.play(FadeIn(backdrop), FadeIn(sub), run_time=0.5)
        self.add(num)
        self.play(counter.animate.set_value(60000), run_time=2.4, rate_func=rush_from)
        self.wait(0.3)
        self.play(FadeOut(num), FadeOut(sub), FadeOut(backdrop), tiles.animate.set_opacity(0.18), run_time=0.6)
        title = T("Day 3/100", 84, WHITE, weight=BOLD).move_to([0, 0.6, 0])
        subt = T("Explaining Deep Learning Concepts", 40, BLUE).next_to(title, DOWN, buff=0.35)
        self.play(FadeIn(title, shift=UP * 0.3), run_time=0.8)
        self.play(FadeIn(subt, shift=UP * 0.2), run_time=0.7)


# ============================================================================= S2  What we're building  (0:08 - 0:56)
class S2_Build(Day3Scene):
    TARGET = 48.0

    def build(self):
        # --- a. the black box: 28x28 image in, 10 scores out ------------------------------
        d = img(SEVEN, 3.4).move_to([-4.6, 0.5, 0])
        grid = pixel_grid(d)
        l1 = T("28 × 28 image", 30).next_to(d, DOWN, buff=0.3)
        l2 = T("= 784 pixels", 30, YELLOW).next_to(l1, DOWN, buff=0.15)
        self.play(FadeIn(d), run_time=0.8)
        self.play(LaggedStart(*[Create(g) for g in grid], lag_ratio=0.02), run_time=1.2)
        self.play(Write(l1), run_time=0.8)
        self.play(Write(l2), run_time=0.8)

        box = RoundedRectangle(corner_radius=0.25, width=3.0, height=2.0, stroke_color=BLUE, stroke_width=3,
                               fill_color=BLUE, fill_opacity=0.12).move_to([0.0, 0.5, 0])
        box_t = VGroup(T("Neural", 32, BLUE), T("network", 32, BLUE)).arrange(DOWN, buff=0.08).move_to(box)
        a_in = Arrow([-2.7, 0.5, 0], [-1.7, 0.5, 0], buff=0.05, color=WHITE)
        a_out = Arrow([1.7, 0.5, 0], [2.7, 0.5, 0], buff=0.05, color=WHITE)
        raw = [0.10, 0.25, 0.15, 0.20, 0.10, 0.12, 0.05, 0.95, 0.30, 0.15]
        stack = VGroup()
        for k, v in enumerate(raw):
            bar = Rectangle(width=1.7 * v + 0.05, height=0.24, stroke_width=0, fill_opacity=1,
                            fill_color=GREEN if k == 7 else GRAY_C)
            lab = T(str(k), 18, WHITE)
            row = VGroup(lab, bar).arrange(RIGHT, buff=0.15)
            stack.add(row)
        stack.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([5.0, 0.5, 0])
        self.play(GrowFromCenter(box), FadeIn(box_t), GrowArrow(a_in), run_time=1.0)
        self.play(GrowArrow(a_out), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in stack], lag_ratio=0.1), run_time=1.4)
        out_cap = T("10 scores, one per digit", 24, GRAY_C).next_to(stack, DOWN, buff=0.3)
        self.play(FadeIn(out_cap), run_time=0.6)
        win = T("highest score = our answer: 7", 28, GREEN).move_to([2.4, -2.9, 0])
        self.play(Indicate(stack[7], color=GREEN, scale_factor=1.15), FadeIn(win), run_time=1.2)
        self.wait(0.8)

        # --- b. open the box: 784 -> 128 -> 64 -> 10 --------------------------------------
        net = Net()
        self.play(FadeOut(VGroup(box, box_t, a_in, a_out, stack, out_cap, win, l1, l2)), FadeOut(grid), run_time=0.7)
        self.play(d.animate.scale(0.5).move_to([-6.1, net.cy, 0]), run_time=0.9)
        self.play(*[FadeIn(g) for g in net.nodes], FadeIn(net.dots),
                  *[Create(e, lag_ratio=0.01) for e in net.edges], run_time=2.6)
        self.play(*[FadeIn(lb) for lb in net.labels], run_time=0.8)
        self.play(LaggedStart(*[FadeIn(p, shift=DOWN * 0.15) for p in net.pills], lag_ratio=0.3), run_time=1.8)

        for l in range(3):  # signal flows forward
            flash = [ShowPassingFlash(e.copy().set_stroke(YELLOW, 2.5, 1), time_width=0.7) for e in net.edges[l]]
            self.play(*flash, run_time=0.8)
        bars = VGroup()
        for k, v in enumerate(raw):
            c = net.pos[3][k]
            bars.add(Rectangle(width=1.9 * v + 0.04, height=0.3, stroke_width=0, fill_opacity=1,
                               fill_color=GREEN if k == 7 else GRAY_C).move_to(c + RIGHT * 0.4, aligned_edge=LEFT))
        self.play(*[GrowFromEdge(b, LEFT) for b in bars], run_time=1.0)
        ans = T("answer: 7", 30, GREEN).move_to([5.75, net.cy - 3.05, 0])
        self.play(Indicate(bars[7], color=GREEN, scale_factor=1.1), FadeIn(ans), run_time=1.0)

        # --- c. what the data fixes vs what we choose -------------------------------------
        sw1 = Square(0.22, stroke_width=0, fill_color=BLUE, fill_opacity=1)
        sw2 = Square(0.22, stroke_width=0, fill_color=YELLOW, fill_opacity=1)
        leg = VGroup(VGroup(sw1, T("fixed by the data", 20)).arrange(RIGHT, buff=0.15),
                     VGroup(sw2, T("our design choice", 20)).arrange(RIGHT, buff=0.15)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-5.6, -1.7, 0]).align_to([-7.0, 0, 0], LEFT)
        self.play(Indicate(net.numlabels[0], color=BLUE), Indicate(net.numlabels[3], color=BLUE), FadeIn(leg[0]), run_time=1.3)
        self.play(Indicate(net.numlabels[1], color=YELLOW), Indicate(net.numlabels[2], color=YELLOW), FadeIn(leg[1]), run_time=1.3)
        self.wait(0.6)

        # --- d. count the parameters ------------------------------------------------------
        self.play(FadeOut(Group(d, leg, bars, ans)),
                  net.body().animate.scale(0.6).move_to([-3.9, 0.3, 0]), run_time=1.3)
        head = T("Counting the parameters", 30, GRAY_C)
        rows = VGroup()
        for tag, f in [("Linear 1", "784 × 128 + 128 = 100,480"), ("Linear 2", "128 × 64 + 64 = 8,256"),
                       ("Linear 3", "64 × 10 + 10 = 650")]:
            rows.add(VGroup(T(tag, 26, BLUE), M(f, 24)).arrange(RIGHT, buff=0.4))
        sep = Line(LEFT * 3.4, RIGHT * 3.4, color=GRAY_C, stroke_width=2)
        total = T("109,386 parameters", 42, YELLOW, weight=BOLD)
        note = T("weights + biases, all random at the start", 24, GRAY_C)
        panel = VGroup(head, rows, sep, total, note).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to([3.1, 0.3, 0])
        self.play(FadeIn(head), run_time=0.6)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.3), run_time=1.3)
        self.play(Create(sep), run_time=0.4)
        self.play(Write(total), run_time=1.2)
        self.play(FadeIn(note), run_time=0.8)
        self.wait(0.5)

        # --- e. Linear = fully connected --------------------------------------------------
        self.play(FadeOut(panel), run_time=0.6)
        one = [ln for (i, j, ln) in net.edge_map[1] if i == 1]
        src = net.nodes[1][1]
        t1 = T("Linear = fully connected layer", 38, YELLOW, weight=BOLD)
        t2 = T("every neuron connects to", 28)
        t3 = T("every neuron in the next layer", 28)
        txt = VGroup(t1, t2, t3).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.1, 0.3, 0])
        self.play(FadeIn(t1), *[ln.animate.set_stroke(YELLOW, 3, 1) for ln in one], src.animate.set_fill(YELLOW, 0.7),
                  run_time=1.3)
        self.play(FadeIn(t2), FadeIn(t3), run_time=1.0)


# ============================================================================= S3  Setup and data  (0:56 - 1:36)
class S3_Data(Day3Scene):
    TARGET = 40.0

    def build(self):
        # --- a. seed + device -------------------------------------------------------------
        box, rows = code_panel(["torch.manual_seed(42)",
                                'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'],
                               [0, 2.85, 0], 22, {"torch": BLUE, "cuda": GREEN, "cpu": ORANGE_C})
        self.play(FadeIn(box), FadeIn(rows), run_time=0.8)
        hl = SurroundingRectangle(rows[0], buff=0.07, color=YELLOW, stroke_width=2)
        self.play(Create(hl), run_time=0.4)
        run = lambda n: VGroup(T(f"run {n}:", 24, GRAY_C), M("tensor([0.3367, 0.1288, 0.2345])", 22)).arrange(RIGHT, buff=0.3)
        r1, r2 = run(1), run(2)
        same = T("same random start every time  →  reproducible", 26, GREEN)
        left = VGroup(r1, r2, same).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-3.3, 0.2, 0])
        self.play(FadeIn(r1, shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeIn(r2, shift=RIGHT * 0.2), run_time=0.7)
        self.play(FadeIn(same), run_time=0.7)
        hl2 = SurroundingRectangle(rows[1], buff=0.07, color=YELLOW, stroke_width=2)
        self.play(Transform(hl, hl2), run_time=0.5)
        cpu = VGroup(*[Square(0.55, stroke_color=ORANGE_C, fill_color=ORANGE_C, fill_opacity=0.5) for _ in range(4)])
        cpu.arrange_in_grid(2, 2, buff=0.1)
        gpu = VGroup(*[Square(0.2, stroke_width=1, stroke_color=GREEN, fill_color=GREEN, fill_opacity=0.5) for _ in range(64)])
        gpu.arrange_in_grid(8, 8, buff=0.05)
        cpu_l = VGroup(cpu, T("CPU: a few fast cores", 20, ORANGE_C)).arrange(DOWN, buff=0.2)
        gpu_l = VGroup(gpu, T("GPU: thousands of small cores", 20, GREEN)).arrange(DOWN, buff=0.2)
        hw = VGroup(cpu_l, gpu_l).arrange(RIGHT, buff=0.7, aligned_edge=DOWN).move_to([3.8, 0.0, 0])
        self.play(FadeIn(hw), run_time=1.0)
        mm = T("matrix multiplication of many numbers → GPU", 24, YELLOW).move_to([3.4, -2.2, 0])
        self.play(FadeIn(mm), Indicate(gpu_l, color=GREEN), run_time=1.3)
        self.wait(0.8)

        # --- b. pixels 0-255 -> 0-1 -------------------------------------------------------
        self.play(FadeOut(Group(left, hw, mm, hl)), run_time=0.5)
        box2, rows2 = code_panel(["transform = transforms.ToTensor()"], [0, 2.85, 0], 22, {"transforms": BLUE})
        self.play(Transform(box, box2), Transform(rows, rows2), run_time=0.6)

        k = 6
        best, bp = -1, (0, 0)
        for r in range(0, 28 - k + 1):
            for c in range(0, 28 - k + 1):
                p = SEVEN[r:r + k, c:c + k]
                sc = int((p > 0).sum()) if (p == 0).sum() >= 8 else -1
                if sc > best:
                    best, bp = sc, (r, c)
        patch = SEVEN[bp[0]:bp[0] + k, bp[1]:bp[1] + k]

        def grid_of(vals, fmt, center):
            cells, nums = VGroup(), VGroup()
            for v in vals.flatten():
                g = int(v)
                sq = Square(0.62, stroke_color=GRAY_C, stroke_width=1, fill_color=rgb_to_color([g / 255] * 3), fill_opacity=1)
                cells.add(sq)
                nums.add(T(fmt(v), 17, BLACK if g > 128 else WHITE))
            cells.arrange_in_grid(k, k, buff=0)
            for n, s in zip(nums, cells):
                n.move_to(s)
            return VGroup(cells, nums).move_to(center)

        gl = grid_of(patch, lambda v: str(int(v)), [-3.9, 0.0, 0])
        gr = grid_of(patch, lambda v: f"{v / 255:.2f}", [3.9, 0.0, 0])
        cap_l = T("pixel value: 0 (black) … 255 (white)", 24, GRAY_C).next_to(gl, DOWN, buff=0.35)
        self.play(FadeIn(gl), run_time=1.0)
        self.play(FadeIn(cap_l), run_time=0.7)
        arr = Arrow([-1.5, 0, 0], [1.5, 0, 0], buff=0, color=YELLOW)
        arr_t = T("ToTensor()   ÷ 255", 26, YELLOW).next_to(arr, UP, buff=0.15)
        self.play(GrowArrow(arr), FadeIn(arr_t), run_time=0.9)
        self.play(FadeIn(gr[0]), LaggedStart(*[TransformFromCopy(a, b) for a, b in zip(gl[1], gr[1])], lag_ratio=0.03),
                  run_time=2.4)
        cap_r = T("0 … 1  →  stable training", 24, GREEN).next_to(gr, DOWN, buff=0.35)
        self.play(FadeIn(cap_r), run_time=0.8)
        self.wait(1.2)

        # --- c. train / test --------------------------------------------------------------
        self.play(FadeOut(Group(gl, gr, cap_l, cap_r, arr, arr_t)), run_time=0.5)
        box3, rows3 = code_panel(['train = datasets.MNIST("data", train=True,  download=True, transform=tf)',
                                  'test  = datasets.MNIST("data", train=False, download=True, transform=tf)'],
                                 [0, 2.85, 0], 20, {"datasets": BLUE, "True": GREEN, "False": RED})
        self.play(Transform(box, box3), Transform(rows, rows3), run_time=0.6)
        tr_bar = Rectangle(width=9.0, height=0.9, stroke_width=0, fill_color=BLUE, fill_opacity=0.85).move_to([-2.0, -0.4, 0])
        te_bar = Rectangle(width=1.5, height=0.9, stroke_width=0, fill_color=ORANGE_C, fill_opacity=0.85).move_to([4.05, -0.4, 0])
        tr_t = T("train: 60,000 images", 28, WHITE, weight=BOLD).move_to(tr_bar)
        te_t = T("test: 10,000", 22, BLACK, weight=BOLD).move_to(te_bar)
        tr_imgs = Group(*[img(XTR[i], 0.9) for i in RNG.choice(60000, 8, replace=False)]).arrange(RIGHT, buff=0.2).next_to(tr_bar, UP, buff=0.35).align_to(tr_bar, LEFT)
        te_imgs = Group(*[img(XTE[i], 0.9) for i in (5, 9)]).arrange(RIGHT, buff=0.2).next_to(te_bar, UP, buff=0.35)
        self.play(GrowFromEdge(tr_bar, LEFT), FadeIn(tr_t), FadeIn(tr_imgs), run_time=1.2)
        self.play(GrowFromEdge(te_bar, LEFT), FadeIn(te_t), FadeIn(te_imgs), run_time=1.0)
        n1 = T("the model learns from these", 26, BLUE).next_to(tr_bar, DOWN, buff=0.35)
        n2 = VGroup(T("kept aside:", 24, ORANGE_C), T("images it has never seen", 24, ORANGE_C)).arrange(DOWN, buff=0.08)
        n2.next_to(te_bar, DOWN, buff=0.35).align_to([6.9, 0, 0], RIGHT)
        self.play(FadeIn(n1), run_time=0.8)
        self.play(FadeIn(n2), run_time=1.0)
        self.wait(1.5)

        # --- d. DataLoader ----------------------------------------------------------------
        self.play(FadeOut(Group(tr_bar, te_bar, tr_t, te_t, tr_imgs, te_imgs, n1, n2)), run_time=0.5)
        box4, rows4 = code_panel(["train_loader = DataLoader(train, batch_size=64, shuffle=True)"],
                                 [0, 2.85, 0], 22, {"DataLoader": BLUE, "True": GREEN, "64": ORANGE_C})
        self.play(Transform(box, box4), Transform(rows, rows4), run_time=0.6)
        big = T("60,000 images", 40, YELLOW, weight=BOLD).move_to([-4.6, 0.7, 0])
        ar = Arrow([-2.4, 0.7, 0], [-1.0, 0.7, 0], buff=0, color=WHITE)
        ar_t = T("÷ 64", 28, WHITE).next_to(ar, UP, buff=0.1)
        cards = VGroup()
        for lab in ["batch 1", "batch 2", "batch 3", "…", "batch 938"]:
            if lab == "…":
                cards.add(T("…", 40, GRAY_C))
            else:
                r = RoundedRectangle(corner_radius=0.12, width=1.55, height=0.95, stroke_color=BLUE, stroke_width=2,
                                     fill_color=BLUE, fill_opacity=0.15)
                cards.add(VGroup(r, T(lab, 20, WHITE).move_to(r)))
        cards.arrange(RIGHT, buff=0.22).move_to([3.0, 0.7, 0])
        self.play(FadeIn(big), run_time=0.6)
        self.play(GrowArrow(ar), FadeIn(ar_t), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in cards], lag_ratio=0.25), run_time=1.6)
        cnt = ValueTracker(0)
        ups = always_redraw(lambda: T(f"{int(cnt.get_value())} small updates", 38, GREEN, weight=BOLD).move_to([0, -1.5, 0]))
        self.add(ups)
        self.play(cnt.animate.set_value(938), run_time=2.2, rate_func=smooth)
        why = T("instead of one giant update on all 60,000 images", 26, GRAY_C).move_to([0, -2.4, 0])
        self.play(FadeIn(why), run_time=0.8)


# ============================================================================= S4  The model  (1:36 - 2:11)
CODE_MODEL = [
    "class DigitNet(nn.Module):",
    "    def __init__(self):",
    "        super().__init__()",
    "        self.layers = nn.Sequential(",
    "            nn.Flatten(),",
    "            nn.Linear(784, 128),",
    "            nn.ReLU(),",
    "            nn.Linear(128, 64),",
    "            nn.ReLU(),",
    "            nn.Linear(64, 10),",
    "        )",
]


class S4_Model(Day3Scene):
    TARGET = 35.0

    def build(self):
        # --- a. Flatten: 28x28 -> 784 numbers ---------------------------------------------
        head = T("The network takes one long list of numbers", 32).move_to([0, 3.2, 0])
        d = img(SEVEN, 3.0).move_to([0, 0.9, 0])
        self.play(FadeIn(head), FadeIn(d), run_time=1.0)
        h = 3.0 / 28
        rows = Group(*[img(SEVEN[i:i + 1, :], 3.0).move_to([0, 0.9 + 1.5 - (i + 0.5) * h, 0]) for i in range(28)])
        self.remove(d)
        self.add(rows)
        sw = 12.0 / 28
        self.play(LaggedStart(*[r.animate.stretch_to_fit_width(sw).stretch_to_fit_height(0.6)
                                .move_to([-6 + (i + 0.5) * sw, -1.2, 0]) for i, r in enumerate(rows)],
                              lag_ratio=0.05), run_time=3.5)
        fl = pill("nn.Flatten()", YELLOW, 28).move_to([0, 0.8, 0])
        dbl = DoubleArrow([-6, -2.0, 0], [6, -2.0, 0], buff=0, color=GRAY_C, stroke_width=3)
        n784 = T("784 numbers  (28 × 28)", 30, YELLOW).next_to(dbl, DOWN, buff=0.15)
        self.play(FadeIn(fl, shift=DOWN * 0.2), run_time=0.8)
        self.play(GrowFromCenter(dbl), FadeIn(n784), run_time=1.0)
        self.wait(0.9)
        self.play(FadeOut(Group(head, rows, fl, dbl, n784)), run_time=0.5)

        # --- b. the class, line by line ---------------------------------------------------
        t2c = {"class": PURPLE_C, "def": PURPLE_C, "nn.Flatten": BLUE, "nn.Linear": BLUE, "nn.ReLU": ORANGE_C,
               "nn.Sequential": BLUE, "nn.Module": BLUE, "784": YELLOW, "128": YELLOW, "64": YELLOW, "10": YELLOW}
        code = code_rows(CODE_MODEL, 21, t2c).move_to([-3.7, 0.0, 0])
        frame = SurroundingRectangle(code, buff=0.3, color=GRAY_C, stroke_width=1.5, corner_radius=0.15)
        self.play(FadeIn(frame), LaggedStart(*[FadeIn(r) for r in code], lag_ratio=0.1), run_time=1.8)

        def focus(i, anim_extra=()):
            rc = SurroundingRectangle(code[i], buff=0.07, color=YELLOW, stroke_width=2)
            return rc

        rc = focus(5)
        self.play(Create(rc), run_time=0.4)
        a1 = VGroup(VGroup(T("784", 34, BLUE, weight=BOLD), T("input pixels: fixed by the data", 24)).arrange(RIGHT, buff=0.3),
                    VGroup(T("128", 34, YELLOW, weight=BOLD), T("neurons: our design choice", 24)).arrange(RIGHT, buff=0.3)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([3.7, 0.5, 0])
        self.play(FadeIn(a1[0], shift=LEFT * 0.2), run_time=0.8)
        self.play(FadeIn(a1[1], shift=LEFT * 0.2), run_time=0.8)
        self.wait(0.9)

        rc2 = focus(6)
        ax = Axes(x_range=[-3, 3, 1], y_range=[-0.5, 3, 1], x_length=3.4, y_length=2.1, tips=False,
                  axis_config={"include_ticks": False, "include_numbers": False, "color": GRAY_C})
        relu = ax.plot(lambda x: max(0.0, x), x_range=[-3, 3, 0.05], color=ORANGE_C, stroke_width=5)
        relu_g = VGroup(ax, relu, T("max(0, z)", 24, ORANGE_C).next_to(ax, UP, buff=0.1)).move_to([3.7, 1.5, 0])
        why = VGroup(T("without ReLU, all the layers", 24), T("collapse into one linear function", 24)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([3.7, -0.9, 0])
        self.play(Transform(rc, rc2), FadeOut(a1), run_time=0.6)
        self.play(FadeIn(relu_g), run_time=0.9)
        self.play(FadeIn(why), run_time=0.9)
        self.wait(0.8)

        rc3 = focus(9)
        a3 = VGroup(T("64 inputs from the hidden layer", 26), T("→  10 final scores", 30, GREEN, weight=BOLD)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.7, 0.3, 0])
        self.play(Transform(rc, rc3), FadeOut(relu_g), FadeOut(why), run_time=0.6)
        self.play(FadeIn(a3), run_time=0.9)
        self.wait(0.8)

        # --- c. forward --------------------------------------------------------------------
        self.play(FadeOut(Group(code, frame, rc, a3)), run_time=0.5)
        spec = [("Flatten", "28×28 → 784", GRAY_C), ("Linear", "784 → 128", BLUE), ("ReLU", "", ORANGE_C),
                ("Linear", "128 → 64", BLUE), ("ReLU", "", ORANGE_C), ("Linear", "64 → 10", BLUE)]
        boxes = VGroup()
        for name, sub, col in spec:
            r = RoundedRectangle(corner_radius=0.15, width=1.9, height=1.1, stroke_color=col, stroke_width=2.5,
                                 fill_color=col, fill_opacity=0.15)
            lab = VGroup(T(name, 24, col, weight=BOLD), T(sub, 17, WHITE)).arrange(DOWN, buff=0.06).move_to(r)
            boxes.add(VGroup(r, lab))
        boxes.arrange(RIGHT, buff=0.3).move_to([0, 0.5, 0])
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.03, stroke_width=3,
                                max_tip_length_to_length_ratio=0.5, color=GRAY_C) for i in range(5)])
        fcode = code_rows(["def forward(self, x):", "    return self.layers(x)"], 26, {"def": PURPLE_C, "return": PURPLE_C}).move_to([0, -1.7, 0])
        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.12), FadeIn(arrows), run_time=1.4)
        self.play(FadeIn(fcode), run_time=0.7)
        dot = Dot(boxes[0].get_left() + LEFT * 0.4, color=YELLOW, radius=0.14)
        self.play(FadeIn(dot), run_time=0.3)
        for b in boxes:
            self.play(dot.animate.move_to(b.get_center() + DOWN * 0.0), Indicate(b[0], color=YELLOW, scale_factor=1.06), run_time=0.45)
        self.play(dot.animate.shift(RIGHT * 1.2), run_time=0.3)
        mem = T("PyTorch remembers every step", 30, YELLOW).move_to([0, -3.0, 0])
        mem2 = T("so backpropagation can replay it later", 24, GRAY_C).next_to(mem, DOWN, buff=0.12)
        self.play(FadeIn(mem), FadeIn(mem2), run_time=1.0)


# ============================================================================= S5  Loss function  (2:11 - 2:47)
LOGITS = np.array([-1.0, 0.3, 0.9, 0.2, -0.6, 0.1, -1.3, 2.4, 0.7, 0.9])
PROBS = np.exp(LOGITS) / np.exp(LOGITS).sum()


class S5_Loss(Day3Scene):
    TARGET = 36.0

    def build(self):
        Y0, BX = -0.2, [-4.5 + i for i in range(10)]

        def bars_from(values, scale, colors=None, fmt="{:.1f}"):
            g, labs, digs = VGroup(), VGroup(), VGroup()
            for i, v in enumerate(values):
                h = max(abs(v) * scale, 0.02)
                col = GREEN if i == 7 else BLUE
                r = Rectangle(width=0.62, height=h, stroke_width=0, fill_color=col, fill_opacity=0.9)
                r.move_to([BX[i], Y0 + (h / 2 if v >= 0 else -h / 2), 0])
                g.add(r)
                lab = T(fmt.format(v), 20, WHITE)
                lab.next_to(r, UP if v >= 0 else DOWN, buff=0.08)
                labs.add(lab)
                digs.add(T(str(i), 24, GRAY_C).move_to([BX[i], -2.3, 0]))
            return g, labs, digs

        # --- a. raw scores -> one number --------------------------------------------------
        title = T("Model output: 10 raw scores", 32).move_to([0, 3.2, 0])
        rb, rl, dg = bars_from(LOGITS, 0.9)
        self.play(FadeIn(title), FadeIn(dg), run_time=0.7)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(rb, LOGITS)], lag_ratio=0.08),
                  FadeIn(rl), run_time=1.6)
        q = T("good or bad? we need to measure it", 28, GRAY_C).move_to([0, -3.2, 0])
        self.play(FadeIn(q), run_time=0.7)
        self.wait(0.5)
        raw_all = Group(rb, rl, dg)
        loss_box = RoundedRectangle(corner_radius=0.2, width=3.2, height=1.3, stroke_color=RED, stroke_width=3,
                                    fill_color=RED, fill_opacity=0.12).move_to([1.6, 0.4, 0])
        loss_t = T("loss function", 30, RED, weight=BOLD).move_to(loss_box)
        a1 = Arrow([-1.2, 0.4, 0], [-0.05, 0.4, 0], buff=0, color=WHITE)
        a2 = Arrow([3.3, 0.4, 0], [4.2, 0.4, 0], buff=0, color=WHITE)
        lv = ValueTracker(2.31)
        lnum = always_redraw(lambda: T(f"{lv.get_value():.2f}", 56, RED, weight=BOLD).move_to([5.7, 0.4, 0]))
        self.play(raw_all.animate.scale(0.42).move_to([-5.2, 0.4, 0]), FadeOut(title), FadeOut(q), run_time=1.0)
        self.play(GrowArrow(a1), FadeIn(loss_box), FadeIn(loss_t), run_time=0.8)
        self.play(GrowArrow(a2), FadeIn(lnum), run_time=0.7)
        one = T("ONE number: how wrong the model is", 30, WHITE).move_to([1.0, -1.4, 0])
        self.play(FadeIn(one), run_time=0.7)
        goal = T("training goal: make it smaller", 30, GREEN, weight=BOLD).move_to([1.0, -2.4, 0])
        self.play(FadeIn(goal), lv.animate.set_value(0.05), run_time=2.2, rate_func=smooth)
        self.wait(0.3)
        self.play(FadeOut(Group(loss_box, loss_t, a1, a2, lnum, one, goal)), run_time=0.5)
        lv.clear_updaters()

        # --- b. softmax -------------------------------------------------------------------
        t_sm = T("Softmax: exp(each score) ÷ sum of all exp(scores)", 30, YELLOW).move_to([0, 3.2, 0])
        self.play(raw_all.animate.scale(1 / 0.42).move_to(Group(rb, rl, dg).get_center() * 0 + [0, 0, 0]), FadeIn(t_sm), run_time=1.0)
        pb, pl, _ = bars_from(PROBS, 5.0, fmt="{:.2f}")
        self.play(*[ReplacementTransform(a, b) for a, b in zip(rb, pb)], FadeOut(rl), run_time=1.6)
        self.play(FadeIn(pl), run_time=0.6)
        sm = T("probabilities: every value between 0 and 1,  sum = 1", 28, GREEN).move_to([0, -3.2, 0])
        self.play(FadeIn(sm), run_time=0.9)
        self.play(Indicate(pb[7], color=WHITE, scale_factor=1.1), Indicate(pl[7]), run_time=1.0)
        self.wait(1.0)
        self.play(FadeOut(Group(t_sm, pb, pl, dg, sm)), run_time=0.5)

        # --- c. -log(p) -------------------------------------------------------------------
        ax = Axes(x_range=[0, 1.0, 0.25], y_range=[0, 5.2, 1], x_length=6.6, y_length=4.4, tips=False,
                  axis_config={"include_numbers": False, "include_ticks": False, "color": GRAY_C}).move_to([-2.6, -0.1, 0])
        curve = ax.plot(lambda x: -np.log(x), x_range=[0.008, 1.0, 0.005], color=YELLOW, stroke_width=5)
        xl = T("p = probability of the correct digit", 22, GRAY_C).next_to(ax, DOWN, buff=0.25)
        yl = T("loss = −log(p)", 24, YELLOW).next_to(ax, UP, buff=0.15).align_to(ax, LEFT)
        ticks = VGroup(*[T(s, 18, GRAY_C).next_to(ax.c2p(v, 0), DOWN, buff=0.1) for v, s in [(0, "0"), (0.5, "0.5"), (1, "1")]],
                       *[T(str(v), 18, GRAY_C).next_to(ax.c2p(0, v), LEFT, buff=0.12) for v in range(1, 6)])
        self.play(Create(ax), FadeIn(ticks), FadeIn(xl), FadeIn(yl), run_time=1.4)
        self.play(Create(curve), run_time=1.6)
        pg, pr = ax.c2p(0.9, -np.log(0.9)), ax.c2p(0.01, -np.log(0.01))
        dg_, dr_ = Dot(pg, color=GREEN, radius=0.12), Dot(pr, color=RED, radius=0.12)
        tg = VGroup(T("p = 0.9", 30, GREEN, weight=BOLD), T("loss ≈ 0.11", 30, GREEN), T("confident and right", 22, GRAY_C)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.4, 1.3, 0])
        tr = VGroup(T("p = 0.01", 30, RED, weight=BOLD), T("loss ≈ 4.61", 30, RED), T("confident and wrong", 22, GRAY_C)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([4.4, -1.1, 0]).align_to(tg, LEFT)
        self.play(FadeIn(dg_, scale=2), FadeIn(tg), run_time=1.2)
        self.play(FadeIn(dr_, scale=2), FadeIn(tr), run_time=1.2)
        pun = T("confidently wrong  →  huge punishment", 30, YELLOW, weight=BOLD).move_to([3.2, -2.9, 0])
        self.play(Indicate(dr_, color=RED, scale_factor=1.8), FadeIn(pun), run_time=1.2)
        self.wait(1.2)
        self.play(FadeOut(Group(ax, curve, xl, yl, ticks, dg_, dr_, tg, tr, pun)), run_time=0.5)

        # --- d. CrossEntropyLoss does both ------------------------------------------------
        chain = VGroup(*[pill(s, c, 26) for s, c in [("10 raw scores", BLUE), ("softmax", YELLOW),
                                                      ("−log(p of correct digit)", YELLOW), ("loss", RED)]])
        chain.arrange(RIGHT, buff=0.55).move_to([0, 0.9, 0])
        arrs = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.06, stroke_width=3,
                              max_tip_length_to_length_ratio=0.5, color=GRAY_C) for i in range(3)])
        wrap = DashedVMobject(SurroundingRectangle(VGroup(chain[1], chain[2]), buff=0.3, color=GREEN, corner_radius=0.2))
        wl = M("nn.CrossEntropyLoss()", 28, GREEN).next_to(wrap, UP, buff=0.2)
        note = T("so the model ends with raw scores: no Softmax layer needed", 28, WHITE).move_to([0, -1.6, 0])
        self.play(LaggedStart(*[FadeIn(c) for c in chain], lag_ratio=0.2), FadeIn(arrs), run_time=1.4)
        self.play(Create(wrap), FadeIn(wl), run_time=1.1)
        self.play(FadeIn(note), run_time=0.9)


# ============================================================================= S6  Wrap-up  (2:47 - 2:57)
class S6_Wrap(Day3Scene):
    TARGET = 10.0

    def build(self):
        head = T("Today: the architecture", 46, WHITE, weight=BOLD).move_to([0, 2.6, 0])
        items = VGroup(*[VGroup(T("✓", 30, GREEN, weight=BOLD), T(s, 28)).arrange(RIGHT, buff=0.3)
                         for s in ["MNIST images → tensors in batches", "784 → 128 → 64 → 10 with ReLU",
                                   "CrossEntropyLoss = softmax + −log"]])
        items.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 0.7, 0])
        self.play(FadeIn(head), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(i, shift=RIGHT * 0.3) for i in items], lag_ratio=0.35), run_time=2.0)
        nxt = VGroup(T("Next video", 30, BLUE, weight=BOLD),
                     T("loss optimization  ·  training cycle  ·  evaluation", 30)).arrange(DOWN, buff=0.15).move_to([0, -1.5, 0])
        self.play(FadeIn(nxt, shift=UP * 0.2), run_time=1.0)
        self.wait(1.0)
        gh = VGroup(T("PyTorch code:", 26, GRAY_C), M("github.com/bishnt/deeplearning-concepts", 26, YELLOW),
                    T("(link in bio)", 22, GRAY_C)).arrange(DOWN, buff=0.12).move_to([0, -3.0, 0])
        self.play(FadeIn(gh), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(Group(head, items, nxt, gh)), run_time=0.5)
        ty = T("Thank you!", 72, WHITE, weight=BOLD).move_to([0, 0.2, 0])
        self.play(FadeIn(ty, scale=0.9), run_time=0.7)
