# Pronunciation Backend MVP

Backend-only MVP for word-level American English pronunciation assessment.

## Scope

- single-word assessment only
- target word known in advance
- `en-US` canonical pronunciation only
- phoneme-level scoring
- one primary issue in the response
- IPA and reference-audio metadata returned with each result

## Architecture

The backend follows the plan's align-based pipeline:

1. `LexiconService` resolves the canonical word entry.
2. `AudioPrepService` decodes audio and computes quality metrics.
3. `SSLFeatureEncoder` produces frame-level speech features.
4. `ConstrainedPhonemeAligner` aligns the audio to the known phone sequence.
5. `PhoneScoringHead` computes `match`, `duration`, `presence`, and `confidence`.
6. `PronunciationPipeline` calibrates scores and returns the API response.

This repository ships a runnable MVP implementation with:

- a deterministic fallback encoder/scorer that works without ML weights
- hooks for a frozen `HuBERT` or `Wav2Vec2` encoder
- resource manifests for a starter `en-US` vocabulary
- training artifact schemas for aligned phoneme supervision

## Run

```bash
pip install -e .[dev]
uvicorn pronunciation_backend.main:app --reload
```

If you also want to experiment with a Hugging Face speech backbone:

```bash
pip install -e .[ml]
```

Set `PRONUNCIATION_BACKBONE_ID` to a compatible checkpoint such as a HuBERT or Wav2Vec2 model. By default the service stays on the lightweight fallback path until a model is available.

## API

`POST /v1/pronunciation/score`

Multipart form fields:

- `word`: target word displayed to the learner
- `audio`: mono recording file
- `speaker_id`: optional

The response returns:

- overall score
- phoneme spans and scores
- one primary issue
- IPA
- reference-audio metadata

## Starter Resources

The bundled lexicon lives in `src/pronunciation_backend/resources/en_us_words.json`.

It is intentionally small and uses one canonical `en-US` pronunciation per word to avoid ambiguous scoring in the MVP.
