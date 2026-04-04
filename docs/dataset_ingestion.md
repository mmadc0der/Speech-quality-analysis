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

The orchestrator also maintains a machine-local dataset cache map at:

```text
<repo>/.pronunciation_dataset_map.json
```

This file is intentionally git-ignored. It records discovered raw roots, requested parts, stage status, and useful handoff paths so reruns can refresh state without rescanning everything manually.

Supported orchestrator datasets:

- `libritts`
- `speechocean762`
- `l2_arctic`
- `librispeech`

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

### Support Matrix

- `libritts`: full orchestrator support for download, raw placement, prepare, and aligned-artifact building once MFA `TextGrid` files and `CMUdict` are available.
- `speechocean762`: local-source import into canonical raw layout, existing prepare reuse, MFA corpus scaffolding reuse, and aligned-artifact building when `TextGrid` files are available.
- `l2_arctic`: first-pass adapter for local-source import plus normalized raw placement only; prepare and align stages are not wired yet.
- `librispeech`: first-pass adapter for direct OpenSLR download plus normalized raw placement only; prepare and align stages are not wired yet.
- feature-store planning and feature precompute remain reusable after aligned manifests exist.

## Unified Orchestrator

The repository now includes:

`python -m pronunciation_backend.training.ingest_datasets`

This command:

1. downloads or imports requested dataset parts into the canonical dataset layout
2. normalizes raw placement when archives contain wrapper directories
3. updates `.pronunciation_dataset_map.json` after every stage boundary
4. reuses the existing `prepare_*`, `build_*_aligned`, and feature-store entrypoints instead of duplicating them

Common stage values:

- `download`
- `prepare`
- `align`
- `feature-plan`
- `feature-precompute`
- `refresh-map`

### Examples

Download all official `LibriTTS` parts and stop after raw placement:

```bash
python -m pronunciation_backend.training.ingest_datasets \
  --datasets libritts \
  --stages download
```

Download one `LibriTTS` subset and immediately prepare manifests:

```bash
python -m pronunciation_backend.training.ingest_datasets \
  --datasets libritts \
  --parts libritts:train-clean-100 \
  --stages download prepare
```

Import a local `SpeechOcean762` copy, prepare manifests, and scaffold MFA inputs:

```bash
python -m pronunciation_backend.training.ingest_datasets \
  --datasets speechocean762 \
  --stages download prepare align \
  --source speechocean762:core=/data/speechocean762
```

Stage a manual `L2-ARCTIC` download into canonical raw placement:

```bash
python -m pronunciation_backend.training.ingest_datasets \
  --datasets l2_arctic \
  --stages download \
  --source l2_arctic:full=/data/L2-ARCTIC
```

Refresh the dataset map without downloading again:

```bash
python -m pronunciation_backend.training.ingest_datasets \
  --datasets libritts speechocean762 l2_arctic librispeech \
  --stages refresh-map
```

## `LibriTTS` Prepare Command

The repository now includes:

`python -m pronunciation_backend.training.prepare_libritts`

The orchestrator calls this command for `libritts` when `prepare` is requested, passing:

- `--dataset-root <dataset>/raw`
- `--output-dir <dataset>/prepared`

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

The orchestrator calls this for `libritts` when `align` is requested and both of the following are supplied:

- `--libritts-textgrid-root`
- `--libritts-cmudict-path`

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

## `SpeechOcean762` Prepare And Align Reuse

The repository already includes:

- `python -m pronunciation_backend.training.prepare_speechocean762`
- `python -m pronunciation_backend.training.prepare_speechocean762_mfa`
- `python -m pronunciation_backend.training.build_speechocean762_aligned`

The orchestrator reuses them in two steps:

1. `prepare` writes `prepared/train.jsonl`, `val.jsonl`, and `test.jsonl`
2. `align` always materializes an MFA-ready mirrored corpus under `<dataset>/reports/mfa_corpus`, and then builds aligned artifacts if `--speechocean-textgrid-root` is supplied

If `--speechocean-textgrid-root` is omitted, the dataset map records the MFA corpus location and leaves the align stage in a scaffolded state.

## Recommended Persistent Setup

For each dataset:

1. download or copy corpus files into `raw/`
2. generate `prepared/*.jsonl`
3. generate `aligned/*.jsonl`
4. run feature precompute

With the orchestrator, the same flow becomes:

1. run `ingest_datasets` with `download`
2. optionally rerun with `prepare`
3. optionally rerun with `align`
4. optionally rerun with `feature-plan` or `feature-precompute`
5. inspect `.pronunciation_dataset_map.json` for the latest discovered paths and stage status

This gives you restartable, inspectable artifacts at every stage instead of a single opaque preprocessing job.
