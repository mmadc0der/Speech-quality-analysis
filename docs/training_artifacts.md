# Training Artifacts

This backend separates runtime inference from training artifacts so the phoneme scorer can evolve without changing the API contract.

For v1 training, treat the speech backbone as an offline feature extractor:

- align utterances first
- run a frozen `HuBERT` or `Wav2Vec2` encoder offline
- materialize per-phone feature rows to disk
- train the scorer as an independent model over cached artifacts

This keeps training cheap on a single consumer GPU and avoids coupling scorer experiments to live waveform encoding.

## Artifact Flow

1. Raw corpora are normalized into dataset-specific `prepared` manifests.
2. Curated word inventory resolves `word -> canonical phones -> IPA -> reference audio`.
3. Native `en-US` and labeled learner recordings are aligned to phoneme spans.
4. A frozen speech backbone extracts frame embeddings offline.
5. Frame embeddings are pooled into per-phone rows and written as cached training artifacts.
6. A standalone scorer trains from cached phone rows without re-running the backbone each epoch.

## Primary Schemas

The canonical Pydantic schemas live in `src/pronunciation_backend/training/schemas.py`.

### `TrainingUtteranceArtifact`

Stores one aligned recording and its supervision payload:

- `utterance_id`
- `speaker_id`
- `dataset`
- `split`
- `target_word`
- `canonical_phones`
- `ipa`
- `audio_path`
- `sample_rate`
- `duration_ms`
- `audio_quality`
- `alignment_source`
- `phone_labels`

### `TrainingPhoneLabel`

Stores one phoneme segment and its human or derived label:

- `phoneme`
- `index`
- `start_ms`
- `end_ms`
- `pronunciation_class`
- `human_score`
- `omission_label`
- `pronounced_phone`

### `PhoneEmbeddingArtifact`

Stores one cached model-ready row for standalone scorer training:

- `speaker_id`
- `dataset`
- `split`
- `target_word`
- `accent_target`
- `prev_phoneme` / `next_phoneme`
- `frame_count`
- `backbone_id`
- `embedding_source`
- pooled phone embedding
- segment variance
- duration and duration z-score
- alignment confidence
- mean segment energy
- raw `human_score`
- class target
- regression target
- omission target

The important design choice is that this artifact is no longer just an internal tensor snapshot. It is the canonical interface between:

- offline feature extraction
- scorer training
- scorer evaluation
- calibration

That means the scorer can be implemented as a completely separate PyTorch module that consumes cached rows from parquet, Arrow, or sharded `.pt` files.

## Suggested Datasets

- native `en-US` read speech such as `LibriTTS` clean subsets for duration priors, canonical embedding calibration, and native false-positive checks
- learner speech with phoneme-level labels, such as `speechocean762`, for scorer supervision

Recommended v1 mix:

- `speechocean762` as the primary supervised scorer dataset
- `LibriTTS` as the primary native reference dataset

Use Hugging Face streaming only for initial ingestion if convenient. For repeated model training, materialize local aligned artifacts and cached phone embeddings first.

## Score Mapping

For datasets that use a `0-2` phoneme label scale:

- `2.0 -> correct`
- `1.0 -> accented`
- `0.0 -> wrong_or_missed`

A simple regression target can be derived as:

- `0.0 -> 15`
- `1.0 -> 60`
- `2.0 -> 92`

This preserves headroom for calibration on held-out `en-US` reference speech.

## Training Split Policy

All scorer experiments should use speaker-disjoint splits.

- `train`: scorer fitting
- `val`: threshold selection, early stopping, and calibration fitting
- `test`: final held-out reporting

Native reference data should also keep a small held-out split so you can measure how often the scorer falsely flags good `en-US` phones.

## Scorer Ownership

The v1 scorer is an independent model, not a jointly trained wrapper around the frozen encoder.

Recommended responsibilities:

- frozen backbone: feature extraction only
- standalone scorer: phone class, `match_score`, and `presence_score`
- duration priors: derive `duration_score`
- calibration layer: map raw outputs into stable runtime scores

This project can still revisit LoRA later, but LoRA is intentionally out of scope for the first cached-feature training baseline.

## Runtime Compatibility

The runtime response contract is intentionally smaller than the training schema:

- runtime returns calibrated phoneme scores and one primary issue
- training stores richer supervision and alignment metadata

That separation lets you retrain or replace the scorer without breaking clients.
