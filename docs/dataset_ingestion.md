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
- `LibriTTS` aligned-artifact builder from MFA `TextGrid` outputs plus `CMUdict`
- helper script for parallel `LibriTTS` MFA alignment launches

### Not implemented yet

- `speechocean762` prepared-manifest builder

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
  --progress-every 5000 \
  --overwrite
```

If you keep `LibriTTS` directly under `/cold/pronunciation/datasets/libritts`, you can also run:

```bash
python -m pronunciation_backend.training.prepare_libritts \
  --dataset-root /cold/pronunciation/datasets/libritts \
  --progress-every 5000 \
  --overwrite
```

The command prints periodic scan progress with:

- processed file count
- prepared row count
- missing transcript count
- ETA in seconds

## `LibriTTS` Aligned Command

The repository now also includes:

`python -m pronunciation_backend.training.build_libritts_aligned`

This command expects:

- `prepared/train.jsonl`, `val.jsonl`, `test.jsonl`
- MFA-generated `TextGrid` files mirrored to the dataset audio paths
- a `CMUdict` file for canonical phone lookup

It emits word-level `TrainingUtteranceArtifact` rows under:

- `aligned/train.jsonl`
- `aligned/val.jsonl`
- `aligned/test.jsonl`
- `aligned/summary.json`

Example:

```bash
python -m pronunciation_backend.training.build_libritts_aligned \
  --dataset-root /cold/pronunciation/datasets/libritts \
  --prepared-dir /cold/pronunciation/datasets/libritts/prepared \
  --output-dir /cold/pronunciation/datasets/libritts/aligned \
  --textgrid-root /cold/pronunciation/datasets/libritts/mfa \
  --cmudict-path /cold/pronunciation/resources/cmudict/cmudict-0.7b \
  --progress-every 250 \
  --overwrite
```

The command prints periodic progress with:

- processed prepared utterances
- emitted aligned word rows
- utterances per second
- ETA in seconds

## Parallel MFA Helper

The repository now includes:

`scripts/run_mfa_parallel_align.sh`

This helper is for the Linux GPU server workflow where you want to launch multiple
long-running MFA alignment jobs with `nohup`.

It exists to avoid a real MFA race condition: if two fresh `mfa align` processes
start at the same time with the same acoustic model alias, they can both try to
unpack the shared model cache under `~/Documents/MFA/extracted_models`, which can
fail with `FileExistsError`.

The helper avoids that by:

- writing `.lab` sidecars from `*.normalized.txt`
- starting the first alignment job alone
- waiting until MFA's shared acoustic-model cache exists
- starting the remaining subsets in parallel after the cache is ready

Example:

```bash
bash scripts/run_mfa_parallel_align.sh train-clean-360 test-clean
```

The script uses these environment variables when you want to override defaults:

- `MFA_BIN`
- `RAW_ROOT`
- `MFA_ROOT`
- `LOG_ROOT`
- `MFA_CACHE_ROOT`

Defaults match the `/cold` layout used by this project.

## Recommended Persistent Setup

For each dataset:

1. download or copy corpus files into `raw/`
2. generate `prepared/*.jsonl`
3. generate `aligned/*.jsonl`
4. run feature precompute

This gives you restartable, inspectable artifacts at every stage instead of a single opaque preprocessing job.
