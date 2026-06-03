"""Synthesize realistic training logs for PhonemeScorerModelV2 and render PNG + TXT.

Log lines mirror ``train_scorer_v2.py`` ``_log(...)`` output (step logs, epoch
summaries, checkpoint messages). Loss curves are plausible reconstructions — no
real run logs exist in the repository.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent
TXT_PATH = DOCS / "scorer_v2_training_logs.txt"
PNG_PATH = DOCS / "scorer_v2_training_logs.png"

EPOCHS = 10
BATCH_SIZE = 128
FREEZE_ENCODER_EPOCHS = 2
OMISSION_LOSS_WEIGHT = 0.25
LOG_EVERY = 100
STEPS_PER_EPOCH = 312
VAL_STEPS = 36
TRAIN_WORDS = STEPS_PER_EPOCH * BATCH_SIZE
VAL_WORDS = VAL_STEPS * BATCH_SIZE
TOKENS_PER_WORD = 4
CKPT_DIR = "/cold/checkpoints/scorer_v2"
TRAIN_FEATS = "/cold/speech_quality/feature_store/v2/splits/train/parquet/words.parquet"
VAL_FEATS = "/cold/speech_quality/feature_store/v2/splits/val/parquet/words.parquet"
ENCODER_CKPT = "/cold/checkpoints/acoustic_encoder_v2/pretrain_best.pt"

rng = random.Random(20240517)


@dataclass
class EpochMetrics:
    epoch: int
    train_quality: float
    train_omit: float
    train_objective: float
    val_quality: float
    val_omit: float
    val_objective: float


def jitter(value: float, frac: float) -> float:
    return value * (1.0 + rng.uniform(-frac, frac))


# --- Synthesized per-epoch loss levels (epochs 1..10) ---------------------------
# These are hand-tuned reconstructions (no real run logs exist) chosen to show a
# realistic mild-overfitting story on the PRIMARY metrics (quality / objective):
#   * TRAIN quality decays quickly, then PLATEAUS (flattens ~0.40 from epoch ~7).
#   * VAL quality tracks train early, then the train/val GAP WIDENS over time.
#   * A localized VAL-only "hill" appears at epochs 7-8 (val rises while train is
#     already flat), then val partially settles but stays clearly above train.
# Because the "best" checkpoint tracks val quality, the hill means epochs 7-8 do
# NOT beat the epoch-6 best; the next best only arrives at epoch 9 (overfit tell).
# Omission / MAE / accuracy follow coherent but gentler trends (no strong hill).
TRAIN_QUALITY = [0.9150, 0.7750, 0.6600, 0.5750, 0.5100, 0.4620, 0.4300, 0.4120, 0.4040, 0.4000]
VAL_QUALITY   = [0.9450, 0.8100, 0.7050, 0.6300, 0.5800, 0.5470, 0.5800, 0.5570, 0.5240, 0.5250]
TRAIN_OMIT    = [0.4600, 0.3800, 0.3200, 0.2750, 0.2400, 0.2050, 0.1800, 0.1600, 0.1480, 0.1400]
VAL_OMIT      = [0.4800, 0.4020, 0.3450, 0.3020, 0.2700, 0.2400, 0.2280, 0.2170, 0.1980, 0.1920]
TRAIN_MAE     = [0.2480, 0.2050, 0.1750, 0.1500, 0.1300, 0.1120, 0.1000, 0.0940, 0.0910, 0.0900]
VAL_MAE       = [0.2620, 0.2200, 0.1900, 0.1660, 0.1460, 0.1300, 0.1360, 0.1300, 0.1210, 0.1210]
TRAIN_ACC     = [0.6100, 0.6650, 0.7100, 0.7450, 0.7750, 0.8000, 0.8200, 0.8320, 0.8380, 0.8400]
VAL_ACC       = [0.5940, 0.6450, 0.6850, 0.7150, 0.7420, 0.7620, 0.7450, 0.7520, 0.7680, 0.7660]


def build_log() -> tuple[str, list[EpochMetrics]]:
    lines: list[str] = []
    metrics: list[EpochMetrics] = []

    lines.append(f"Loaded pretrained acoustic encoder from {ENCODER_CKPT}")
    lines.append(f"train: using dense parquet dataset from {TRAIN_FEATS}")
    cw = [round(jitter(1.71, 0.02), 6), round(jitter(0.83, 0.02), 6), round(jitter(0.46, 0.02), 6)]
    lines.append(f"train: class weights={cw}")
    lines.append(f"val: using dense parquet dataset from {VAL_FEATS}")
    val_words_per_s = jitter(9800.0, 0.05)
    lines.append(
        f"val: cached {VAL_STEPS} batch(es) / {VAL_WORDS} words in CPU RAM "
        f"({val_words_per_s:.1f} words/s)"
    )

    best_val_quality = float("inf")

    for epoch in range(EPOCHS):
        lines.append(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        if epoch == FREEZE_ENCODER_EPOCHS:
            lines.append("encoder: trainable")

        q_level = TRAIN_QUALITY[epoch]
        o_level = TRAIN_OMIT[epoch]
        mae_level = TRAIN_MAE[epoch]
        wps_level = 5400.0 if epoch < FREEZE_ENCODER_EPOCHS else 3150.0

        for step in range(LOG_EVERY, STEPS_PER_EPOCH + 1, LOG_EVERY):
            prog = step / STEPS_PER_EPOCH
            q = jitter(q_level * (1.06 - 0.10 * prog), 0.05)
            o = jitter(o_level * (1.05 - 0.08 * prog), 0.06)
            mae = jitter(mae_level * (1.05 - 0.07 * prog), 0.05)
            wps = jitter(wps_level, 0.04)
            lines.append(
                f"Train Step {step:05d} | "
                f"Quality L: {q:.4f} | "
                f"Omit L: {o:.4f} | "
                f"Score MAE: {mae:.4f} | "
                f"{wps:.1f} words/s"
            )

        tr_q = jitter(q_level, 0.008)
        tr_o = jitter(o_level, 0.008)
        tr_mae = jitter(mae_level, 0.008)
        tr_acc = jitter(TRAIN_ACC[epoch], 0.004)
        tr_obj = tr_q + OMISSION_LOSS_WEIGHT * tr_o
        lines.append(
            f"Train Summary | Epoch {epoch + 1} | "
            f"Steps: {STEPS_PER_EPOCH} | "
            f"Words: {TRAIN_WORDS} | "
            f"Tokens: {TRAIN_WORDS * TOKENS_PER_WORD} | "
            f"Quality: {tr_q:.4f} | "
            f"Omit: {tr_o:.4f} | "
            f"Score MAE: {tr_mae:.4f} | "
            f"Class Acc: {tr_acc:.4f} | "
            f"Objective: {tr_obj:.4f}"
        )

        va_q = jitter(VAL_QUALITY[epoch], 0.008)
        va_o = jitter(VAL_OMIT[epoch], 0.008)
        va_mae = jitter(VAL_MAE[epoch], 0.008)
        va_acc = jitter(VAL_ACC[epoch], 0.005)
        va_obj = va_q + OMISSION_LOSS_WEIGHT * va_o
        lines.append(
            f"Val Summary   | Epoch {epoch + 1} | "
            f"Steps: {VAL_STEPS} | "
            f"Words: {VAL_WORDS} | "
            f"Tokens: {VAL_WORDS * TOKENS_PER_WORD} | "
            f"Quality: {va_q:.4f} | "
            f"Omit: {va_o:.4f} | "
            f"Score MAE: {va_mae:.4f} | "
            f"Class Acc: {va_acc:.4f} | "
            f"Objective: {va_obj:.4f}"
        )

        metrics.append(
            EpochMetrics(
                epoch=epoch + 1,
                train_quality=tr_q,
                train_omit=tr_o,
                train_objective=tr_obj,
                val_quality=va_q,
                val_omit=va_o,
                val_objective=va_obj,
            )
        )

        lines.append(f"Saved checkpoint to {CKPT_DIR}/scorer_v2_epoch_{epoch + 1}.pt")
        if va_q < best_val_quality:
            best_val_quality = va_q
            lines.append(
                f"New best validation checkpoint saved to {CKPT_DIR}/scorer_v2_best.pt "
                f"(quality_loss={best_val_quality:.4f})"
            )

    return "\n".join(lines) + "\n", metrics


def _terminal_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/lucon.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _color_for(line: str) -> tuple[int, int, int]:
    if line.startswith("--- Epoch"):
        return (88, 166, 255)
    if line.startswith("New best"):
        return (63, 185, 80)
    if line.startswith("Saved checkpoint"):
        return (210, 168, 255)
    if line.startswith("Train Summary"):
        return (126, 231, 135)
    if line.startswith("Val Summary"):
        return (255, 196, 76)
    if line.startswith(("Train Step", "Val Step")):
        return (139, 148, 158)
    if line.startswith("encoder:"):
        return (255, 123, 114)
    return (201, 209, 217)


def render_terminal_panel(text: str, *, scale: int = 2) -> Image.Image:
    lines = text.splitlines()
    font = _terminal_font(15 * scale // 2)
    pad = 22 * scale // 2
    line_h = 22 * scale // 2
    tmp = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(tmp)
    max_w = max(draw.textlength(ln, font=font) for ln in lines)
    width = int(max_w) + 2 * pad
    header_h = 38 * scale // 2
    height = header_h + line_h * len(lines) + 2 * pad

    img = Image.new("RGB", (width, height), (13, 17, 23))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, header_h], fill=(22, 27, 34))
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = pad + 12 + i * 22
        cy = header_h // 2
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    title = "train_scorer_v2.py — синтезированный журнал"
    draw.text((pad + 96, header_h // 2 - 9), title, font=font, fill=(139, 148, 158))

    y = header_h + pad
    for ln in lines:
        draw.text((pad, y), ln, font=font, fill=_color_for(ln))
        y += line_h
    return img


def _setup_cyrillic_font() -> None:
    for candidate in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            return


def render_loss_panel(
    metrics: list[EpochMetrics], *, height_px: int, dpi: int = 120
) -> Image.Image:
    _setup_cyrillic_font()
    epochs = [m.epoch for m in metrics]

    # Width is chosen wide enough that the combined figure (log panel + graph
    # panel) is comfortably non-square; height matches the terminal panel so the
    # stacked plots fill the full vertical extent with no top/bottom whitespace.
    width_in = 12.0
    height_in = height_px / dpi
    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi, facecolor="#f6f8fa")
    gs = gridspec.GridSpec(
        3, 1, figure=fig, hspace=0.30, top=0.955, bottom=0.045, left=0.085, right=0.975
    )

    series = [
        ("Потери качества", "train_quality", "val_quality"),
        ("Потери пропусков", "train_omit", "val_omit"),
        ("Сводная цель обучения", "train_objective", "val_objective"),
    ]
    for ax_idx, (title, train_key, val_key) in enumerate(series):
        ax = fig.add_subplot(gs[ax_idx])
        train_y = [getattr(m, train_key) for m in metrics]
        val_y = [getattr(m, val_key) for m in metrics]
        ax.plot(epochs, train_y, "o-", color="#2f80ed", linewidth=2.2, markersize=7, label="обучение")
        ax.plot(epochs, val_y, "s--", color="#e67e22", linewidth=2.2, markersize=7, label="валидация")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("эпоха")
        ax.set_ylabel("потери")
        ax.set_xticks(epochs)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right", fontsize=9)
        if ax_idx == 0:
            ax.annotate(
                "СИНТЕЗ",
                xy=(0.02, 0.96),
                xycoords="axes fraction",
                fontsize=9,
                color="#666",
                alpha=0.85,
            )

    fig.suptitle("Динамика потерь по эпохам", fontsize=15, fontweight="bold", y=0.992)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    plot_img = Image.frombytes("RGBA", (w, h), bytes(buf)).convert("RGB")
    plt.close(fig)
    return plot_img


def render_combined_png(text: str, metrics: list[EpochMetrics]) -> None:
    terminal = render_terminal_panel(text)
    plots = render_loss_panel(metrics, height_px=terminal.height)

    gap = 24
    total_w = terminal.width + gap + plots.width
    total_h = max(terminal.height, plots.height)
    canvas = Image.new("RGB", (total_w, total_h), (246, 248, 250))
    canvas.paste(terminal, (0, (total_h - terminal.height) // 2))
    canvas.paste(plots, (terminal.width + gap, (total_h - plots.height) // 2))
    canvas.save(PNG_PATH, optimize=True)


def main() -> None:
    text, metrics = build_log()
    TXT_PATH.write_text(text, encoding="utf-8")
    render_combined_png(text, metrics)
    size = PNG_PATH.stat().st_size
    print(f"wrote {TXT_PATH} ({len(text.splitlines())} lines)")
    print(f"wrote {PNG_PATH} ({size} bytes)")


if __name__ == "__main__":
    main()
