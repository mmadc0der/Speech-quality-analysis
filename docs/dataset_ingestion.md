# Dataset Ingestion

## Decision

Use a preload-first dataset strategy.

For this project, persistent local copies are better than training directly from Hugging Face streaming:

- reproducible paths across runs
- easier restart behavior on a single GPU server
- clearer separation between raw, prepared, aligned, and feature caches
- less dependence on network stability during long preprocessing jobs

Hugging Face can still be used as the download source, but after download the project should operate on local files under `/cold`.

## Canonical Dataset Layout

Each dataset should live under:

```text
/cold/pronunciation/datasets/<dataset>/
  raw/
  prepared/
  aligned/
  reports/
```

Recommended first datasets:

- `libritts`
- `speechocean762`
- optional later: `l2_arctic`

## Processing Stages

The intended pipeline is:

1. `raw`
2. `prepared`
3. `aligned`
4. `features`

### `raw`

Original downloaded corpus files.

Examples:

- `LibriTTS` subset directories
- `speechocean762` audio and annotation files

### `prepared`

Dataset-specific raw files converted into a common utterance manifest format.

The canonical schema is `PreparedUtteranceArtifact` in `src/pronunciation_backend/training/schemas.py`.

Prepared manifests should be written as:

```text
<dataset-root>/prepared/train.jsonl
<dataset-root>/prepared/val.jsonl
<dataset-root>/prepared/test.jsonl
```

### `aligned`

Prepared utterances converted into the project's scorer-facing training format:

- canonical phones resolved
- alignment spans available
- phone labels available or derived

Aligned manifests should be written as:

```text
<dataset-root>/aligned/train.jsonl
<dataset-root>/aligned/val.jsonl
<dataset-root>/aligned/test.jsonl
```

Each line must validate as `TrainingUtteranceArtifact`.

### `features`

Frozen backbone outputs pooled into `PhoneEmbeddingArtifact` rows and written into the hashed feature store under `/cold/pronunciation/features`.

## Current Support

### Implemented now

- hashed feature-store planning and verification
- actual feature precompute runner
- persistent `LibriTTS` prepared-manifest builder

### Not implemented yet

- `speechocean762` prepared-manifest builder
- aligned-artifact generation from prepared manifests
- MFA orchestration inside the repo

## `LibriTTS` Prepare Command

The repository now includes:

`python -m pronunciation_backend.training.prepare_libritts`

This scans a preloaded `LibriTTS` root, finds audio files and sibling transcript files, maps subset names to `train / val / test`, and writes:

- `prepared/train.jsonl`
- `prepared/val.jsonl`
- `prepared/test.jsonl`
- `prepared/summary.json`

Example:

```bash
python -m pronunciation_backend.training.prepare_libritts \
  --dataset-root /cold/pronunciation/datasets/libritts/raw \
  --overwrite
```

If you keep `LibriTTS` directly under `/cold/pronunciation/datasets/libritts`, you can also run:

```bash
python -m pronunciation_backend.training.prepare_libritts \
  --dataset-root /cold/pronunciation/datasets/libritts \
  --overwrite
```

## Recommended Persistent Setup

For each dataset:

1. download or copy corpus files into `raw/`
2. generate `prepared/*.jsonl`
3. generate `aligned/*.jsonl`
4. run feature precompute

This gives you restartable, inspectable artifacts at every stage instead of a single opaque preprocessing job.
