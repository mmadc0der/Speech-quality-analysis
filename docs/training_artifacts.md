# Training Artifacts

This backend separates runtime inference from training artifacts so the phoneme-scoring head can evolve without changing the API contract.

## Artifact Flow

1. Curated word inventory resolves `word -> canonical phones -> IPA -> reference audio`.
2. Native `en-US` and labeled learner recordings are aligned to phoneme spans.
3. Frame-level embeddings are extracted from the frozen speech backbone.
4. Per-phone pooled features become training rows for the small scoring head.

## Primary Schemas

The canonical Pydantic schemas live in `src/pronunciation_backend/training/schemas.py`.

### `TrainingUtteranceArtifact`

Stores one aligned recording and its supervision payload:

- `utterance_id`
- `speaker_id`
- `dataset`
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

Stores one model-ready feature row for scorer training:

- pooled phone embedding
- segment variance
- duration and duration z-score
- alignment confidence
- mean segment energy
- class target
- regression target
- omission target

## Suggested Datasets

- native `en-US` read speech for duration priors and canonical embedding calibration
- learner speech with phoneme-level labels, such as `speechocean762`, for scorer supervision

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

## Runtime Compatibility

The runtime response contract is intentionally smaller than the training schema:

- runtime returns calibrated phoneme scores and one primary issue
- training stores richer supervision and alignment metadata

That separation lets you retrain or replace the scorer without breaking clients.
