"""Synthesize acoustic-encoder pretrain logs and render PNG + TXT.

Log lines mirror ``pretrain_acoustic_encoder_v2.py`` ``_log(...)`` output. The
scenario omits a validation split (train-only LibriTTS-style pretrain). Curves are
plausible reconstructions — NO real run logs exist in the repository (SYNTHESIZED).

Modelling note
--------------
The real ``pretrain_acoustic_encoder_v2.py`` optimises a *continuous* masked-feature
reconstruction objective (MSE over 768-dim acoustic embeddings) and has no explicit
discrete codebook. For these slide visuals we present the pretraining objective in
the more familiar masked-TOKEN-prediction framing (cross-entropy / perplexity). We
adopt the standard HuBERT-style acoustic-unit codebook size of VOCAB_SIZE = 504, so
the cross-entropy loss starts near ln(504) ≈ 6.22 (a uniform-prior model) and decays
exponentially as the encoder learns. Perplexity = exp(loss).

The loss is synthesized PER TRAINING STEP:
    loss(step) = final + (initial - final) * exp(-k * progress)
                 + high_frequency_sgd_jitter + heteroscedastic_noise + rare_spikes
where ``initial = ln(VOCAB_SIZE)``, the decay is steep early and flattens later. On
top of the smooth trend we overlay fine-grained, high-frequency step-to-step jitter
(like real mini-batch SGD), and the overall noise amplitude GROWS toward the end of
training (later steps are noisier / more unstable).
"""

from __future__ import annotations

import math
import random

import numpy as np
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent
TXT_PATH = DOCS / "encoder_pretrain_logs.txt"
PNG_PATH = DOCS / "encoder_pretrain_logs.png"

EPOCHS = 10
BATCH_SIZE = 256                           # real default from pretrain_acoustic_encoder_v2.py (--batch-size)
LOG_EVERY = 100                            # real default (--log-every) — the cadence the script actually logs at
DISPLAY_EVERY = 5000                       # panel sampling: show one of the emitted step lines every N steps (slide-readable)
MASK_RATIO = 0.20                          # real default (--mask-ratio)
MASK_BLOCK_SIZE = 2
PHONES_PER_WORD = 4                         # avg phonemes per word unit (token count basis)

# Dataset scale (SYNTHESIZED, LibriTTS-960-scale ESTIMATE).
# The pretrain unit is one WORD (= one utterance_id in the feature store; see
# docs/scorer_v2_training_process_ru.md). build_libritts_aligned.py emits word-level
# rows from LibriTTS train. Published LibriTTS train split sizes (sentence utterances):
#   train-clean-100 = 33,236 + train-clean-360 = 116,500 + train-other-500 = 205,044
#   => 354,780 sentence utterances. At ~16.1 words/utterance this is ~5.71M word units.
# No real run logs exist in the repo, so this is an estimate stated as such.
LIBRITTS_UTTERANCES = 354_780
AVG_WORDS_PER_UTTERANCE = 16.1
NUM_TRAIN_WORDS = int(round(LIBRITTS_UTTERANCES * AVG_WORDS_PER_UTTERANCE))   # ~5,711,958
STEPS_PER_EPOCH = math.ceil(NUM_TRAIN_WORDS / BATCH_SIZE)                     # ceil(units / batch)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
TRAIN_WORDS = NUM_TRAIN_WORDS

CKPT_DIR = "/cold/checkpoints/acoustic_encoder_v2"
CKPT_PREFIX = "acoustic_encoder_v2"
TRAIN_FEATS = "/cold/speech_quality/feature_store/libritts/splits/train/mmap"

# HuBERT-style acoustic-unit codebook (see module docstring). Cross-entropy of a
# uniform-prior model over this many tokens is ln(VOCAB_SIZE).
VOCAB_SIZE = 504
INITIAL_LOSS = math.log(VOCAB_SIZE)        # ≈ 6.2226
FINAL_LOSS = 0.92                          # plateau ≈ ln(504) / 6.8  (perplexity ≈ 2.5)
DECAY_K = 5.0                              # steep early drop, flatten later

rng = random.Random(20240603)


@dataclass
class EpochMetrics:
    epoch: int
    loss: float
    masked_tokens: int


def jitter(value: float, frac: float) -> float:
    return value * (1.0 + rng.uniform(-frac, frac))


def masked_tokens_for_step() -> int:
    # ~20% of tokens in a batch of BATCH_SIZE words × ~PHONES_PER_WORD phonemes/word
    base = int(BATCH_SIZE * PHONES_PER_WORD * MASK_RATIO)
    return int(jitter(float(base), 0.08))


def step_loss(global_step: int) -> float:
    """Per-step cross-entropy loss: exponential decay + SGD-like noise.

    Anchored at ``ln(VOCAB_SIZE)`` for step 0, dropping fast early and flattening.
    Three noise components reproduce a realistic mini-batch SGD loss trace:
      1. high-frequency jitter present on EVERY step (small amplitude),
      2. a heteroscedastic component whose amplitude GROWS toward the end, and
      3. rare larger instability spikes, concentrated late in training.
    """
    progress = global_step / max(1, TOTAL_STEPS - 1)
    trend = FINAL_LOSS + (INITIAL_LOSS - FINAL_LOSS) * math.exp(-DECAY_K * progress)

    # (1) Always-on high-frequency step-to-step jitter (real SGD mini-batch noise).
    hf = rng.gauss(0.0, 0.018 * trend)

    # (2) Heteroscedastic noise: small early, larger late (≈1% → ≈9% of trend).
    noise_frac = 0.010 + 0.080 * (progress ** 1.3)
    het = rng.gauss(0.0, noise_frac * trend)

    # (3) Occasional larger instability spikes, concentrated late in training.
    spike = 0.0
    if rng.random() < 0.04 + 0.10 * progress:
        spike = rng.gauss(0.0, 0.10 * trend * progress)

    return max(0.05, trend + hf + het + spike)


def build_log() -> tuple[str, list[EpochMetrics], list[int], list[float]]:
    # One shared synthetic per-step loss array drives BOTH the text logs and the
    # graph, so printed perplexity (= exp(loss)) always matches the curve.
    curve_steps: list[int] = list(range(1, TOTAL_STEPS + 1))
    curve_loss: list[float] = [step_loss(s) for s in curve_steps]

    lines: list[str] = []
    metrics: list[EpochMetrics] = []

    lines.append(f"train: using mmap dataset from {TRAIN_FEATS}")
    lines.append(
        f"train: pretrain config | objective=masked_token_ce | vocab={VOCAB_SIZE} | "
        f"batch_size={BATCH_SIZE} | mask_ratio={MASK_RATIO} | epochs={EPOCHS} | "
        f"steps/epoch={STEPS_PER_EPOCH} | log_every={LOG_EVERY}"
    )

    # STEP-based log: the script logs a "Train Step" line every LOG_EVERY steps; for a
    # slide-readable panel we sample one of those lines every DISPLAY_EVERY steps. The
    # dominant content is step lines; checkpoints are saved at epoch boundaries.
    wps_level = 6200.0
    next_epoch_boundary = STEPS_PER_EPOCH
    epoch_done = 0
    for global_step in curve_steps:
        if global_step % DISPLAY_EVERY == 0:
            loss = curve_loss[global_step - 1]
            ppl = math.exp(loss)
            masked = masked_tokens_for_step()
            wps = jitter(wps_level, 0.05)
            lines.append(
                f"Train Step {global_step:06d} | "
                f"CE loss: {loss:.4f} | "
                f"ppl: {ppl:.2f} | "
                f"Masked tokens: {masked} | "
                f"{wps:.1f} words/s"
            )

        if global_step >= next_epoch_boundary and epoch_done < EPOCHS:
            epoch_done += 1
            ckpt = f"{CKPT_DIR}/{CKPT_PREFIX}_epoch_{epoch_done}.pt"
            lines.append(f"Saved checkpoint to {ckpt}  (epoch {epoch_done}/{EPOCHS}, step {global_step:06d})")
            metrics.append(
                EpochMetrics(
                    epoch=epoch_done,
                    loss=curve_loss[global_step - 1],
                    masked_tokens=int(TRAIN_WORDS * PHONES_PER_WORD * MASK_RATIO * jitter(1.0, 0.02)),
                )
            )
            next_epoch_boundary += STEPS_PER_EPOCH

    return "\n".join(lines) + "\n", metrics, curve_steps, curve_loss


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
    if line.startswith("Saved checkpoint"):
        return (210, 168, 255)
    if line.startswith("Train Summary"):
        return (126, 231, 135)
    if line.startswith("Train Step"):
        return (139, 148, 158)
    if line.startswith("#"):
        return (110, 118, 129)
    return (201, 209, 217)


def render_terminal_panel(text: str) -> Image.Image:
    lines = text.splitlines()
    font = _terminal_font(15)
    pad = 22
    line_h = 22
    tmp = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(tmp)
    max_w = max(draw.textlength(ln, font=font) for ln in lines)
    width = int(max_w) + 2 * pad
    header_h = 38
    height = header_h + line_h * len(lines) + 2 * pad

    img = Image.new("RGB", (width, height), (13, 17, 23))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, header_h], fill=(22, 27, 34))
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = pad + 12 + i * 22
        cy = header_h // 2
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    draw.text(
        (pad + 96, header_h // 2 - 9),
        "pretrain_acoustic_encoder_v2.py — синтезированный журнал",
        font=font,
        fill=(139, 148, 158),
    )

    y = header_h + pad
    for ln in lines:
        draw.text((pad, y), ln, font=font, fill=_color_for(ln))
        y += line_h
    return img


def _setup_cyrillic_font() -> None:
    for candidate in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            return


def render_loss_panel(
    curve_steps: list[int],
    curve_loss: list[float],
    *,
    height_px: int,
    dpi: int = 120,
) -> Image.Image:
    _setup_cyrillic_font()

    # Subsample the noisy per-step curve for plotting (hundreds of thousands of
    # points); the underlying loss array is unchanged, so logged perplexity stays
    # consistent with the displayed curve.
    stride = max(1, len(curve_steps) // 6000)
    curve_steps = curve_steps[::stride]
    curve_loss = curve_loss[::stride]

    # Smooth exponential trend underneath the noisy per-step curve.
    trend = [
        FINAL_LOSS + (INITIAL_LOSS - FINAL_LOSS) * math.exp(-DECAY_K * (s / max(1, TOTAL_STEPS - 1)))
        for s in curve_steps
    ]

    # Width is chosen wide enough that the combined figure (log panel + graph
    # panel) is comfortably non-square; height matches the terminal panel so the
    # plot fills the full vertical extent with no top/bottom whitespace.
    width_in = 12.0
    height_in = height_px / dpi
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi, facecolor="#f6f8fa")
    ax.plot(
        curve_steps,
        curve_loss,
        color="#7b61ff",
        linewidth=0.9,
        alpha=0.55,
        label="обучение (по шагам)",
    )
    ax.plot(
        curve_steps,
        trend,
        color="#1f6feb",
        linewidth=2.6,
        label="экспон. тренд",
    )

    ax.axhline(INITIAL_LOSS, color="#d29922", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.annotate(
        f"ln(словарь={VOCAB_SIZE}) ≈ {INITIAL_LOSS:.2f}",
        xy=(TOTAL_STEPS * 0.42, INITIAL_LOSS),
        xytext=(TOTAL_STEPS * 0.42, INITIAL_LOSS + 0.05),
        fontsize=9,
        color="#9a6700",
    )

    ax.set_title("Потери предобучения (маскированные токены)", fontsize=13, fontweight="bold")
    ax.set_xlabel("шаг обучения")
    ax.set_ylabel("потери (кросс-энтропия)")
    ax.set_xlim(0, TOTAL_STEPS)
    ax.set_ylim(0, INITIAL_LOSS + 0.6)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")

    # Secondary axis: perplexity = exp(loss).
    def _loss_to_ppl(x):
        return np.exp(np.asarray(x, dtype=float))

    def _ppl_to_loss(x):
        return np.log(np.clip(np.asarray(x, dtype=float), 1e-6, None))

    secax = ax.secondary_yaxis("right", functions=(_loss_to_ppl, _ppl_to_loss))
    secax.set_ylabel("перплексия = exp(потери)")

    ax.annotate(
        f"доля маски {int(MASK_RATIO * 100)}%  |  per-step  |  синтез",
        xy=(0.02, 0.035),
        xycoords="axes fraction",
        fontsize=9,
        color="#666",
    )
    fig.subplots_adjust(top=0.955, bottom=0.075, left=0.085, right=0.9)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    plot_img = Image.frombytes("RGBA", (w, h), bytes(buf)).convert("RGB")
    plt.close(fig)
    return plot_img


def render_combined_png(
    text: str,
    curve_steps: list[int],
    curve_loss: list[float],
) -> None:
    terminal = render_terminal_panel(text)
    plots = render_loss_panel(curve_steps, curve_loss, height_px=terminal.height)
    gap = 24
    total_w = terminal.width + gap + plots.width
    total_h = max(terminal.height, plots.height)
    canvas = Image.new("RGB", (total_w, total_h), (246, 248, 250))
    canvas.paste(terminal, (0, (total_h - terminal.height) // 2))
    canvas.paste(plots, (terminal.width + gap, (total_h - plots.height) // 2))
    canvas.save(PNG_PATH, optimize=True)


def main() -> None:
    text, _metrics, curve_steps, curve_loss = build_log()
    TXT_PATH.write_text(text, encoding="utf-8")
    render_combined_png(text, curve_steps, curve_loss)
    size = PNG_PATH.stat().st_size
    print(f"vocab(codebook)={VOCAB_SIZE}  init={INITIAL_LOSS:.4f}  final≈{curve_loss[-1]:.4f}")
    print(f"wrote {TXT_PATH} ({len(text.splitlines())} lines)")
    print(f"wrote {PNG_PATH} ({size} bytes)")


if __name__ == "__main__":
    main()
