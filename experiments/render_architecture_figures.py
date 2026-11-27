"""Render the architecture and pipeline figures used in docs/13_MODEL_ARCHITECTURES.md.

Run from the repository root:  python experiments/render_architecture_figures.py
Writes five PNG files to docs/figures/ (13a ... 13e). The architecture numbers (stages,
channels, depths, patch grids, token counts) were read from the timm models themselves.

Colours: one hue per model family, used the same way in every figure. The palette's first
three categorical slots were checked with the dataviz validator (colour-blind safe as a set):
ConvNeXt = blue, DINOv2 = orange, DINOv3 = aqua.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

INK, MUTED, GRID, SURFACE = "#191919", "#666666", "#d9d9d9", "#ffffff"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"   # ConvNeXt, DINOv2, DINOv3
NEUTRAL_FILL, NEUTRAL_EDGE = "#f3f3f3", "#9c9c9c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK})


def tint(hex_color: str, amount: float) -> str:
    """Mix a colour with white (amount 0 = white, 1 = the colour)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: round(255 - (255 - c) * amount)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def canvas(width: float, height: float, title: str, subtitle: str = ""):
    figure, axis = plt.subplots(figsize=(width, height))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.axis("off")
    figure.patch.set_facecolor(SURFACE)
    axis.text(0.3, height - 0.35, title, fontsize=15, weight="bold", va="top")
    if subtitle:
        axis.text(0.3, height - 0.8, subtitle, fontsize=10, color=MUTED, va="top")
    return figure, axis


def box(axis, x, y, w, h, text, color=None, fontsize=9.5, bold_first=True, align="center"):
    """Rounded box; first line bold. ``color`` = family colour (light fill, coloured edge)."""
    face = tint(color, 0.14) if color else NEUTRAL_FILL
    edge = color if color else NEUTRAL_EDGE
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                  facecolor=face, edgecolor=edge, linewidth=1.4))
    lines = text.split("\n")
    total = len(lines)
    step = min(0.24, (h - 0.12) / max(total, 1))
    top = y + h / 2 + step * (total - 1) / 2
    for index, line in enumerate(lines):
        tx = x + w / 2 if align == "center" else x + 0.15
        axis.text(tx, top - index * step, line, ha="center" if align == "center" else "left", va="center",
                  fontsize=fontsize + (0.5 if index == 0 and bold_first else 0),
                  weight="bold" if index == 0 and bold_first else "normal",
                  color=INK if index == 0 or not bold_first else "#333333")


def box_both(axis, x, y, w, h, text, fontsize=9):
    """Box for a part shared by DINOv2 and DINOv3: orange edge plus a dashed aqua inner border."""
    box(axis, x, y, w, h, text, color=ORANGE, fontsize=fontsize)
    axis.add_patch(FancyBboxPatch((x + 0.06, y + 0.06), w - 0.12, h - 0.12, boxstyle="round,pad=0,rounding_size=0.1",
                                  facecolor="none", edgecolor=AQUA, linewidth=1.4, linestyle=(0, (3, 2))))


def arrow(axis, x1, y1, x2, y2, color=MUTED, style="-|>"):
    axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                  arrowprops=dict(arrowstyle=style, color=color, lw=1.3, shrinkA=2, shrinkB=2,
                                  mutation_scale=12))


def label(axis, x, y, text, size=9, color=MUTED, ha="center", weight="normal"):
    axis.text(x, y, text, fontsize=size, color=color, ha=ha, va="center", weight=weight)


def legend_row(axis, x, y, items):
    for name, color in items:
        axis.add_patch(FancyBboxPatch((x, y - 0.11), 0.32, 0.22, boxstyle="round,pad=0,rounding_size=0.05",
                                      facecolor=tint(color, 0.14), edgecolor=color, linewidth=1.4))
        axis.text(x + 0.42, y, name, fontsize=9.5, va="center")
        x += 0.42 + len(name) * 0.085 + 0.45


def save(figure, name):
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print("saved", OUT / name)


# --------------------------------------------------------------------------- 13a pipeline
def pipeline():
    f, a = canvas(20.2, 7.2, "The SPAGHETTI-1 pipeline, from raw images to a submission",
                  "Every step is a command of the spaghetti1 package; numbers are from the current state (2026-11-27).")
    y, h = 3.0, 2.6
    columns = [
        ("1  Data", "Kaggle an2dl2627\n4,328 labelled\ntraining images\n14,736 test images\n256 × 256 RGB, 10 classes"),
        ("2  Cleaning", "remove 332 alien overlays\nremove 1 blank frame\n721 duplicate groups:\noriginal + noisy copies\n→ 3,995 training images"),
        ("3  Validation", "5 folds, grouped by\nduplicate group,\nstratified by class\nfold 0: 3,196 train\n599 validation images"),
        ("4  Augmentation", "random rotation 0–360°,\nmirror, scale 0.9–1.1\ncolour casts, brightness,\nnoise, blur, overlays\n→ resize + normalise"),
    ]
    x = 0.3
    for title, body in columns:
        box(a, x, y, 2.55, h, f"{title}\n{body}", fontsize=9)
        arrow(a, x + 2.55, y + h / 2, x + 2.85, y + h / 2)
        x += 2.85
    # models column
    models = [("ConvNeXt-Tiny 224 · TTA", BLUE), ("ConvNeXt-Tiny 384 · TTA", BLUE), ("ConvNeXt-Small 288 · TTA", BLUE),
              ("DINOv2-Base 336", ORANGE), ("DINOv3-Base 384", AQUA)]
    mx, mw, mh = x, 3.0, 0.5
    label(a, mx + mw / 2, y + h + 0.6, "5  Five models (fold 0)", size=10.5, color=INK, weight="bold")
    for index, (name, color) in enumerate(models):
        my = y + h + 0.35 - (index + 1) * 0.68
        box(a, mx, my, mw, mh, name, color=color, fontsize=9)
        arrow(a, mx + mw, my + mh / 2, mx + mw + 0.45, y + h / 2)
    x = mx + mw + 0.45
    box(a, x, y + 0.45, 2.6, 1.7, "6  Ensemble\naverage the 10 class\nprobabilities of\nthe 5 models\n→ highest class", fontsize=9)
    arrow(a, x + 2.6, y + h / 2, x + 2.95, y + h / 2)
    box(a, x + 2.95, y + 0.75, 1.95, 1.1, "submission.csv\nKaggle 0.8441\n(best so far)", fontsize=9)
    label(a, 0.3, 1.6, "TTA = 8 rotated/mirrored views averaged at prediction time (ConvNeXt only; it did not help DINOv2).",
          size=9.5, ha="left")
    label(a, 0.3, 1.15, "Colours mark the model family in every figure:", size=9.5, ha="left")
    legend_row(a, 4.3, 1.15, [("ConvNeXt (ImageNet-22k)", BLUE), ("DINOv2", ORANGE), ("DINOv3", AQUA)])
    save(f, "13a_pipeline.png")


# --------------------------------------------------------------------------- 13b ConvNeXt
def convnext():
    f, a = canvas(17, 9.4, "ConvNeXt: the convolutional network family we used (Tiny, Small, and DINOv3 ConvNeXt-Tiny)",
                  "Four stages; each halves the image size and doubles the channels. Sizes below are for a 224 × 224 input (288 × 288 in brackets).")
    y, h = 5.5, 2.0
    blocks = [
        ("Input image", "3 × 224 × 224", None, 1.6),
        ("Stem", "4×4 conv, stride 4\n96 × 56 × 56\n(96 × 72 × 72)", BLUE, 1.75),
        ("Stage 1", "3 blocks\n96 × 56 × 56\n(× 72 × 72)", BLUE, 1.55),
        ("Stage 2", "3 blocks\n192 × 28 × 28\n(× 36 × 36)", BLUE, 1.55),
        ("Stage 3", "Tiny: 9 blocks\nSmall: 27 blocks\n384 × 14 × 14\n(× 18 × 18)", BLUE, 1.75),
        ("Stage 4", "3 blocks\n768 × 7 × 7\n(× 9 × 9)", BLUE, 1.55),
        ("Head", "global average\npool → 768\nLayerNorm\nLinear 768 → 10", None, 1.75),
        ("Output", "10 class\nprobabilities\n(softmax)", None, 1.45),
    ]
    x = 0.3
    for index, (title, body, color, w) in enumerate(blocks):
        box(a, x, y, w, h, f"{title}\n{body}", color=color, fontsize=9)
        if index < len(blocks) - 1:
            arrow(a, x + w, y + h / 2, x + w + 0.42, y + h / 2)
            if 2 <= index <= 4:
                label(a, x + w + 0.21, y + h + 0.22, "down-\nsample", size=7.5)
        x += w + 0.42
    # block detail
    label(a, 0.3, 4.45, "Inside one ConvNeXt block (repeated in every stage):", size=11, color=INK, ha="left", weight="bold")
    steps = [("7×7 depthwise conv", "each channel looks at\nits 7×7 neighbourhood"),
             ("LayerNorm", "normalise the\nfeatures"),
             ("1×1 conv  C → 4C", "mix channels,\nexpand 4×"),
             ("GELU", "non-linearity"),
             ("1×1 conv  4C → C", "back to C\nchannels"),
             ("layer scale +\ndrop path", "scaled; randomly\nskipped in training")]
    x, by, bw, bh = 1.3, 1.65, 2.05, 1.55
    label(a, 0.55, by + bh / 2, "x", size=12, color=INK, weight="bold")
    arrow(a, 0.7, by + bh / 2, x, by + bh / 2)
    for title, body in steps:
        box(a, x, by, bw, bh, f"{title}\n{body}", color=BLUE, fontsize=8.6)
        arrow(a, x + bw, by + bh / 2, x + bw + 0.3, by + bh / 2)
        x += bw + 0.3
    a.add_patch(plt.Circle((x + 0.25, by + bh / 2), 0.22, facecolor=SURFACE, edgecolor=INK, lw=1.3))
    label(a, x + 0.25, by + bh / 2, "+", size=14, color=INK, weight="bold")
    arrow(a, x + 0.47, by + bh / 2, x + 0.95, by + bh / 2)
    label(a, x + 1.15, by + bh / 2, "out", size=10, color=INK)
    top = by + bh + 0.35   # skip path: up from the input, over all blocks, down into the "+"
    a.plot([0.95, 0.95, x + 0.25], [by + bh / 2, top, top], color=MUTED, lw=1.3)
    arrow(a, x + 0.25, top, x + 0.25, by + bh / 2 + 0.22)
    label(a, (0.95 + x) / 2, top + 0.22, "residual (skip) connection: the input x is added back to the block's output", size=8.5)
    label(a, 0.3, 0.75, "Used: ConvNeXt-Tiny (28 M weights, ImageNet-22k) at 224/288/384 px · ConvNeXt-Small (50 M, same stages, 27 blocks in stage 3) at 288 px · "
                        "DINOv3 ConvNeXt-Tiny (28 M, identical shape, weights distilled from DINOv3).", size=9, ha="left")
    save(f, "13b_convnext.png")


# --------------------------------------------------------------------------- 13c ViT
def vit():
    f, a = canvas(17, 10.4, "Vision Transformer: DINOv2-Base and DINOv3-Base",
                  "The image is cut into patches; every patch becomes a token; 12 transformer blocks let all tokens exchange information.")
    y, h = 6.3, 2.05
    steps = [
        ("Input image", "DINOv2: 336 × 336\nDINOv3: 384 × 384", None, 1.9),
        ("Cut into patches", "24 × 24 = 576 patches\nDINOv2: 14 × 14 px\nDINOv3: 16 × 16 px", None, 2.0),
        ("Patch embedding", "each patch → a vector\nof 768 numbers\n(linear projection)", None, 2.0),
        ("Tokens", "1 CLS token\n+ 4 register tokens\n+ 576 patch tokens\n= 581 × 768", None, 1.95),
        ("12 transformer\nblocks", "12 attention heads\nwidth 768\n(86 M weights)", "both", 2.0),
        ("Summary vector", "DINOv2: CLS token\nDINOv3: average of\npatch tokens", "both", 1.95),
        ("Head", "LayerNorm\nLinear 768 → 10\n→ softmax", None, 1.65),
    ]
    x = 0.3
    for index, (title, body, color, w) in enumerate(steps):
        if color == "both":   # half orange, half aqua edge: shared by both models
            box(a, x, y, w, h, f"{title}\n{body}", color=ORANGE, fontsize=9)
            a.add_patch(FancyBboxPatch((x + 0.06, y + 0.06), w - 0.12, h - 0.12, boxstyle="round,pad=0,rounding_size=0.1",
                                       facecolor="none", edgecolor=AQUA, linewidth=1.4, linestyle=(0, (3, 2))))
            a.texts[-1].set_text(a.texts[-1].get_text())
        else:
            box(a, x, y, w, h, f"{title}\n{body}", fontsize=9)
        if index < len(steps) - 1:
            arrow(a, x + w, y + h / 2, x + w + 0.38, y + h / 2)
        x += w + 0.38
    label(a, 8.0, y - 0.4, "Position information: DINOv2 adds a learned position vector to each patch token; "
                           "DINOv3 uses rotary position encoding (RoPE) inside attention.", size=9)

    # transformer block detail
    label(a, 0.3, 5.0, "Inside one transformer block (× 12):", size=11, color=INK, ha="left", weight="bold")
    box_both(a, 9.2, 4.86, 0.42, 0.28, "", fontsize=1)
    label(a, 9.75, 5.0, "= the same in DINOv2 and DINOv3 (the main computing parts); differences are in the table below",
          size=9, ha="left")
    bx, by, bw, bh = 1.2, 2.55, 2.15, 1.5
    detail = [("LayerNorm", "normalise\ntokens"), ("Multi-head\nself-attention", "every token looks at\nall 581 tokens, 12 heads"),
              ("+ residual", "add the input\nback"), ("LayerNorm", "normalise\nagain"),
              ("MLP", "768 → 3072 → 768\nGELU, per token"), ("+ residual", "add back\n(DINOv2: layer scale)")]
    label(a, 0.55, by + bh / 2, "tokens", size=9.5, color=INK)
    arrow(a, 0.85, by + bh / 2, bx, by + bh / 2)
    for index, (title, body) in enumerate(detail):
        if index in (1, 4):
            box_both(a, bx, by, bw, bh, f"{title}\n{body}", fontsize=8.6)
        else:
            box(a, bx, by, bw, bh, f"{title}\n{body}", fontsize=8.6)
        if index < len(detail) - 1:
            arrow(a, bx + bw, by + bh / 2, bx + bw + 0.28, by + bh / 2)
        bx += bw + 0.28
    arrow(a, bx - 0.28, by + bh / 2, bx + 0.1, by + bh / 2)
    label(a, bx + 0.5, by + bh / 2, "tokens", size=9.5, color=INK)

    # comparison table
    rows = [("", "DINOv2-Base (R2)", "DINOv3-Base (N2)"),
            ("Pretraining", "self-supervised, 142 M images (2023)", "self-supervised, ~1.7 B images (2025)"),
            ("Patch size / input", "14 px / 336 px → 24 × 24", "16 px / 384 px → 24 × 24"),
            ("Position encoding", "learned vector per patch", "rotary (RoPE) in attention"),
            ("Summary for the head", "CLS token", "average of patch tokens"),
            ("Our result (fold 0)", "best single model: 0.852", "0.832, but most robust (Kaggle 0.8253)")]
    tx, ty, col = 0.3, 1.75, [3.0, 4.6, 4.6]
    for r, row in enumerate(rows):
        cx = tx
        for c, cell in enumerate(row):
            color = INK if r == 0 or c == 0 else "#333333"
            if r == 0 and c == 1:
                color = ORANGE
            if r == 0 and c == 2:
                color = "#00866b"
            a.text(cx, ty - r * 0.3, cell, fontsize=9, color=color, weight="bold" if r == 0 or c == 0 else "normal", va="center")
            cx += col[c]
    a.plot([tx, tx + sum(col)], [ty - 0.15, ty - 0.15], color=GRID, lw=1)
    save(f, "13c_vision_transformer.png")


# --------------------------------------------------------------------------- 13d best ensemble
def ensemble():
    f, a = canvas(15, 8.0, "Our best submission: a five-model ensemble (Kaggle public 0.8441)",
                  "Each model sees the same test image at its own input size; the ten class probabilities are averaged with equal weight.")
    members = [("ConvNeXt-Tiny", "224 px · 8-view TTA", "val 0.844", BLUE),
               ("ConvNeXt-Tiny", "384 px · 8-view TTA", "val 0.846", BLUE),
               ("ConvNeXt-Small", "288 px · 8-view TTA", "val 0.847", BLUE),
               ("DINOv2-Base", "336 px · no TTA", "val 0.851", ORANGE),
               ("DINOv3-Base", "384 px · no TTA", "val 0.832 · most robust", AQUA)]
    box(a, 0.4, 3.0, 2.0, 1.4, "Test image\n256 × 256\n(one of 14,736)", fontsize=9.5)
    top = 6.1
    for index, (name, detail, score, color) in enumerate(members):
        y = top - index * 1.05
        arrow(a, 2.4, 3.7, 3.2, y + 0.4)
        box(a, 3.2, y, 4.2, 0.8, f"{name}\n{detail}  ·  {score}", color=color, fontsize=9.2)
        box(a, 7.9, y + 0.1, 1.7, 0.6, "10 probabilities", fontsize=8.5, bold_first=False)
        arrow(a, 7.4, y + 0.4, 7.9, y + 0.4)
        arrow(a, 9.6, y + 0.4, 10.3, 3.7)
    box(a, 10.3, 2.95, 2.0, 1.5, "Average\n(equal weights)\n→ class with the\nhighest probability", fontsize=9)
    arrow(a, 12.3, 3.7, 12.8, 3.7)
    box(a, 12.8, 3.05, 1.9, 1.3, "Prediction\none of 10 classes\n→ submission.csv", fontsize=9)
    label(a, 0.4, 1.25, "Ensemble scores: clean validation 0.855 · test-like validation 0.861 · Kaggle public 0.8441.", size=9.5, ha="left")
    label(a, 0.4, 0.85, "Adding DINOv3 lowered clean validation but raised the Kaggle score (0.8424 → 0.8441):", size=9.5, ha="left")
    label(a, 0.4, 0.45, "its robustness to the test images' corruptions matters more than its clean score.", size=9.5, ha="left")
    save(f, "13d_best_ensemble.png")


# --------------------------------------------------------------------------- 13e scores (dot plot)
def scores():
    rows = [  # (label, family colour, clean val, val + TTA, Kaggle or None)
        ("ConvNeXt-Tiny 224 (E1a)", BLUE, 0.843, 0.844, None),
        ("ConvNeXt-Tiny 288 (E2a)", BLUE, 0.846, 0.833, None),
        ("ConvNeXt-Tiny 384 (E2b)", BLUE, 0.847, 0.846, None),
        ("ConvNeXt-Small 288 (E3)", BLUE, 0.841, 0.847, None),
        ("ConvNeXt-Tiny 224, research recipe (R1)", BLUE, 0.831, 0.828, None),
        ("DINOv3 ConvNeXt-Tiny 288 (N1)", AQUA, 0.840, 0.833, None),
        ("DINOv2-Base 336 (R2)", ORANGE, 0.852, 0.839, 0.8321),
        ("DINOv3-Base 384 (N2)", AQUA, 0.832, 0.843, 0.8253),
    ]
    f, a = plt.subplots(figsize=(11.5, 6.4))
    f.patch.set_facecolor(SURFACE)
    series = [("Clean validation", BLUE, "o", 2), ("Validation + TTA", ORANGE, "D", 3), ("Kaggle public (alone)", AQUA, "s", 4)]
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        values = [v for v in row[2:] if v is not None]
        a.plot([min(values), max(values)], [y, y], color=GRID, lw=2, zorder=1)
        for (name, color, marker, column) in series:
            value = row[column]
            if value is None:
                continue
            a.scatter(value, y, s=70, color=color, marker=marker, edgecolor=SURFACE, linewidth=2, zorder=3,
                      label=name if i == 6 else None)
        best = max(values)
        a.text(best + 0.0012, y, f"{row[2]:.3f}" + (f" · Kaggle {row[4]:.4f}" if row[4] else ""), va="center",
               fontsize=8.5, color=MUTED)
    a.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=9.5)
    for tick, row in zip(a.get_yticklabels(), list(reversed(rows))):
        tick.set_color(INK)
    a.set_xlim(0.818, 0.868)
    a.set_xlabel("Weighted F1 (fold-0 validation: 599 images, about ±1.5 points of noise)", color=MUTED)
    a.grid(axis="x", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    for side in ("top", "right", "left"):
        a.spines[side].set_visible(False)
    a.spines["bottom"].set_color(GRID)
    a.tick_params(axis="y", length=0)
    a.tick_params(axis="x", colors=MUTED)
    a.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=9.5)
    a.set_title("Every architecture we tested (single models, fold 0)", loc="left", fontsize=13, weight="bold", pad=34)
    f.text(0.01, -0.02, "Numbers next to each row: clean validation, and the Kaggle score where the model was submitted alone. "
                        "Best ensemble (5 models): Kaggle 0.8441.", fontsize=8.5, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    f.savefig(OUT / "13e_model_scores.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(f)
    print("saved", OUT / "13e_model_scores.png")


if __name__ == "__main__":
    pipeline()
    convnext()
    vit()
    ensemble()
    scores()
