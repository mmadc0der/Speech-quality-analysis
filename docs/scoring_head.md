# Scoring Head Design

## V1 Decision

The first trainable pronunciation model in this repository is a standalone phoneme scorer trained from cached phone-level artifacts.

For v1:

- keep the speech backbone frozen
- precompute phone-level features offline
- train the scorer independently from cached artifacts
- derive `duration_score` mostly from native duration priors instead of fully supervised learning

This design is optimized for a single mid-range GPU and fast experiment turnover.

## Model Boundary

The scorer does not consume raw audio directly.

Its input is one `PhoneEmbeddingArtifact` row per canonical phoneme segment, produced after:

1. audio preparation
2. constrained alignment
3. frozen backbone feature extraction
4. phone-span pooling

## Input Features

Recommended v1 features per phoneme row:

- `mean_embedding`
- scalar `variance`
- `duration_ms`
- `duration_z_score`
- `alignment_confidence`
- `energy_mean`
- target phoneme identity
- optional previous phoneme identity
- optional next phoneme identity
- optional segment position flags such as word-initial or word-final

## Outputs

The scorer should predict:

- phoneme class: `correct / accented / wrong_or_missed`
- `match_score`
- omission or weak-realization probability

The runtime service then derives:

- `presence_score`
- calibrated `match_score`
- `duration_score`
- `confidence`
- final phoneme `status`

## Suggested Architecture

Recommended baseline:

- phoneme-id embedding table
- shared MLP trunk over concatenated continuous features and phoneme embeddings
- classification head for phoneme class
- regression head for `match_score`
- binary head for omission or weak realization

Example trunk:

- `Linear(input_dim, 512)`
- `LayerNorm(512)`
- `GELU`
- `Dropout(0.2)`
- `Linear(512, 256)`
- `GELU`
- `Dropout(0.1)`
- `Linear(256, 128)`
- `GELU`

Example heads:

- `class_head -> 3 logits`
- `match_head -> 1 scalar`
- `omission_head -> 1 logit`

## Targets

Primary supervision:

- `pronunciation_class`
- `regression_target`
- `omission_target`

Recommended target mapping for `speechocean762`-style `0-2` phone labels:

- `2 -> correct`
- `1 -> accented`
- `0 -> wrong_or_missed`

Suggested regression mapping:

- `0 -> 15`
- `1 -> 60`
- `2 -> 92`

## Losses

Recommended first loss mix:

- weighted cross-entropy for phoneme class
- Huber loss for `match_score`
- BCE with class weighting for omission detection

Illustrative weighting:

- `1.0 * class_loss`
- `0.7 * match_loss`
- `0.5 * omission_loss`

## Dataset Mix

Use two data sources:

- `speechocean762` for supervised pronunciation labels
- native `en-US` reference speech such as `LibriTTS` for duration priors and native false-positive validation

Use speaker-disjoint `train / val / test` splits for scorer experiments.

## Evaluation

Track at least:

- macro F1 for phoneme class
- F1 for `wrong_or_missed`
- Spearman correlation between predicted `match_score` and human phone score
- omission F1
- native false-positive rate
- calibration error after score calibration

## Out Of Scope For V1

The following are intentionally deferred:

- joint end-to-end backbone plus scorer training
- LoRA over the frozen backbone
- sequence models over full utterances
- sentence-level pronunciation assessment
