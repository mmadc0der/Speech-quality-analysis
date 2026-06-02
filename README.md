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
4. `MfaForcedAligner` runs forced alignment against the known transcript in a separate MFA subprocess.
5. `PhoneScoringHead` computes `match`, `duration`, `presence`, and `confidence`.
6. `PronunciationPipeline` calibrates scores and returns the API response.

This repository ships a runnable MVP implementation with:

- a frozen `HuBERT`-style runtime encoder and `v2` scorer path
- MFA-based inference-time phoneme alignment
- resource manifests for curated reference-audio metadata
- training artifact schemas for aligned phoneme supervision

## Training Direction

The current training direction is intentionally split from runtime inference:

1. align labeled utterances to canonical phone spans
2. run a frozen `HuBERT` or `Wav2Vec2` encoder offline
3. cache per-phone feature rows to disk
4. train a standalone phoneme scorer from cached artifacts

That means v1 training does not keep the frozen backbone and the scorer inside one end-to-end trainable module. The backbone acts as an embedding extractor, while the scorer is a separate model trained cheaply on cached phone-level tensors.

The recommended first dataset mix is:

- `speechocean762` for supervised phoneme-quality labels
- native `en-US` read speech such as `LibriTTS` for duration priors and native-reference calibration

For the first training-launch milestone, see `docs/feature_precompute_pipeline.md`. It defines the `/cold` storage layout, hashed feature-cache policy, aligned-artifact expectations, and the actual feature precompute command.

For dataset handling, see `docs/dataset_ingestion.md`. The project now follows a preload-first `raw -> prepared -> aligned -> features` pipeline, starting with a persistent `LibriTTS` prepared-manifest builder.

## Run Backend

Install dependencies (core deps include `torch` and `transformers`; there is no `[ml]` extra):

```bash
uv sync --group dev
uv run uvicorn pronunciation_backend.main:app --reload
```

Or with pip editable install (runtime only; use `uv sync --group dev` for pytest/httpx):

```bash
pip install -e .
uvicorn pronunciation_backend.main:app --reload
```

Set `PRONUNCIATION_BACKBONE_ID` to a compatible Hugging Face checkpoint such as `facebook/hubert-base-ls960` (default).

For the current `scorer_v2` serving path, set the runtime env vars before launch. The backend process stays in the project `uv` environment, while MFA is launched through an explicit external command, typically from micromamba:

```bash
export PRONUNCIATION_USE_HF_ENCODER=1
export PRONUNCIATION_SCORER_CHECKPOINT_PATH=/path/to/scorer_v2_best.pt
export PRONUNCIATION_SCORER_DEVICE=cuda
export PRONUNCIATION_MFA_COMMAND="/opt/micromamba/bin/micromamba run -n mfa mfa"
export PRONUNCIATION_MFA_ACOUSTIC_MODEL=english_us_arpa
uvicorn pronunciation_backend.main:app --host 0.0.0.0 --port 8000
```

Notes:

- `PRONUNCIATION_MFA_COMMAND` should resolve to the `mfa` executable itself or to a launcher command that ends by invoking `mfa`; the backend appends the `align` subcommand and request-specific arguments.
- `PRONUNCIATION_MFA_ACOUSTIC_MODEL` is passed directly to MFA and should point at the acoustic model you want to use for inference-time alignment.
- if MFA is unavailable or misconfigured, the app can still start, but `POST /v1/pronunciation/score` will fail with a clear `503` response

## Run Lightweight Frontend

Run the debug frontend locally on your workstation and point it at a remote backend:

```bash
python -m pronunciation_backend.frontend --backend-url http://your-server:8000 --port 3000
```

Or use environment variables:

```bash
export PRONUNCIATION_FRONTEND_BACKEND_URL=http://your-server:8000
export PRONUNCIATION_FRONTEND_PORT=3000
python -m pronunciation_backend.frontend
```

Then open `http://127.0.0.1:3000`.

The frontend proxies requests through the local app, so the browser does not need direct CORS access to the remote backend.

The debug UI now supports two trim modes for manual verification:

- backend auto-trim: default path; the server detects the likely spoken-word window before scoring
- frontend manual trim: optional local crop by start/end milliseconds before upload

If you enable frontend manual trim and want to prevent the server from re-trimming the already cropped clip, enable the UI option that sends `noTrim=true`.

## API

`POST /v1/pronunciation/score`

Multipart form fields:

- `word`: target word displayed to the learner
- `audio`: mono recording file
- `speaker_id`: optional
- `noTrim`: optional boolean flag that skips backend auto-trim

The backend aligns the trimmed clip with MFA on every scoring request. It does not use the old heuristic phoneme partitioner for runtime scoring.

The response returns:

- overall score
- phoneme spans and scores
- one primary issue
- IPA
- reference-audio metadata

## Runtime Vocabulary

Runtime scoring resolves target words from CMUdict. The bundled file
`src/pronunciation_backend/resources/en_us_words.json` remains a curated override
layer for reference audio, IPA, syllables, and stress metadata on a small starter
set of words.

- CMUdict-backed words score normally and return `reference: null` unless curated.
- Unknown tokens, proper nouns missing from CMUdict, and non-English words still
  return `404`.
- Optional `PRONUNCIATION_CMUDICT_PATH` can pin a local dictionary file; otherwise
  the `cmudict` package supplies the runtime dictionary.
