# Feature Precompute Pipeline

## Goal

The first runnable training stage is offline backbone feature extraction.

This stage should:

1. verify that required datasets are already present on disk
2. derive a deterministic feature-store key from the backbone and preprocessing spec
3. create a dedicated feature directory for that exact spec
4. write manifests so later steps can detect whether features already exist
5. precompute phone-level artifacts into split-specific directories

For v1, all heavyweight artifacts should live under `/cold`.

## Storage Policy

Use `/cold` as the root for caches and training artifacts.

Recommended environment:

```bash
export HF_HOME=/cold/huggingface
export PRONUNCIATION_STORAGE_ROOT=/cold/pronunciation
export PRONUNCIATION_DATASET_ROOT=/cold/pronunciation/datasets
export PRONUNCIATION_FEATURE_ROOT=/cold/pronunciation/features
export PRONUNCIATION_CHECKPOINT_ROOT=/cold/pronunciation/checkpoints
export PRONUNCIATION_REPORT_ROOT=/cold/pronunciation/reports
```

Recommended directory layout:

```text
/cold/
  huggingface/
  pronunciation/
    datasets/
      speechocean762/
      libritts/
      l2_arctic/
    features/
      speechocean762/
        <feature_key>/
          spec.json
          state.json
          logs/
          splits/
            train/
            val/
            test/
      libritts/
        <feature_key>/
          ...
    checkpoints/
    reports/
```

## Feature Key Policy

Each precompute output directory is namespaced by a deterministic hash of the feature spec.

The hash should change when any of the following changes:

- dataset name
- backbone id
- backbone revision
- adapter or LoRA id
- embedding source
- alignment source
- pooling version
- artifact schema version
- sample rate

The hash should not change when you switch from `train val test` to just `train` for a smoke test. Split selection controls which subdirectories are populated, not the cache identity itself.

This lets you safely keep multiple cached feature sets side by side. If you later introduce LoRA or a different pooling method, it automatically lands in a new directory rather than silently mixing artifacts.

## Current Planning Utility

The repository now includes a small planning utility:

`python -m pronunciation_backend.training.feature_store`

Supported commands:

- `plan`: create the hashed feature-store directory and manifests
- `verify`: check dataset presence and feature-store manifests

This utility is the control-plane layer for the precompute runner.

The actual feature extractor now lives at:

`python -m pronunciation_backend.training.precompute_features`

The extractor auto-initializes the feature-store by default, so `feature_store plan` is optional once you move to normal runs.

## Precompute Spec Example

For the first baseline:

- dataset: `speechocean762`
- splits: `train val test`
- backbone: `facebook/hubert-base-ls960`
- backbone revision: `main`
- adapter id: empty
- embedding source: `hubert`
- alignment source: `mfa`
- pooling version: `phone_mean_v1`
- artifact schema version: `phone_embedding_artifact_v1`
- sample rate: `16000`

## Finite Launch Sequence

Use this exact order on the GPU server.

### Step 1. Export storage paths

```bash
export HF_HOME=/cold/huggingface
export PRONUNCIATION_STORAGE_ROOT=/cold/pronunciation
export PRONUNCIATION_DATASET_ROOT=/cold/pronunciation/datasets
export PRONUNCIATION_FEATURE_ROOT=/cold/pronunciation/features
export PRONUNCIATION_CHECKPOINT_ROOT=/cold/pronunciation/checkpoints
export PRONUNCIATION_REPORT_ROOT=/cold/pronunciation/reports
```

### Step 2. Verify datasets are mounted where expected

At minimum for the first precompute pass:

- `/cold/pronunciation/datasets/speechocean762`
- `/cold/pronunciation/datasets/libritts`

Optional later:

- `/cold/pronunciation/datasets/l2_arctic`

### Step 3. Optionally create a hashed feature-store plan for `speechocean762`

```bash
python -m pronunciation_backend.training.feature_store plan \
  --dataset speechocean762 \
  --dataset-root /cold/pronunciation/datasets/speechocean762 \
  --splits train val test \
  --backbone-id facebook/hubert-base-ls960 \
  --backbone-revision main \
  --embedding-source hubert \
  --alignment-source mfa \
  --pooling-version phone_mean_v1 \
  --artifact-schema-version phone_embedding_artifact_v1 \
  --sample-rate 16000
```

This command creates:

- `spec.json`
- `state.json`
- split directories under the hashed feature-store path

### Step 4. Optionally verify the planned feature-store directory

```bash
python -m pronunciation_backend.training.feature_store verify \
  --dataset speechocean762 \
  --dataset-root /cold/pronunciation/datasets/speechocean762 \
  --splits train val test \
  --backbone-id facebook/hubert-base-ls960 \
  --backbone-revision main \
  --embedding-source hubert \
  --alignment-source mfa \
  --pooling-version phone_mean_v1 \
  --artifact-schema-version phone_embedding_artifact_v1 \
  --sample-rate 16000
```

If this passes, the storage contract is ready for the real extractor.

### Step 5. Optionally repeat the same plan for native-reference data

For example:

```bash
python -m pronunciation_backend.training.feature_store plan \
  --dataset libritts \
  --dataset-root /cold/pronunciation/datasets/libritts \
  --splits train val test \
  --backbone-id facebook/hubert-base-ls960 \
  --backbone-revision main \
  --embedding-source hubert \
  --alignment-source mfa \
  --pooling-version phone_mean_v1 \
  --artifact-schema-version phone_embedding_artifact_v1 \
  --sample-rate 16000
```

### Step 6. Make sure aligned artifacts exist

The extractor expects aligned utterance artifacts under:

```text
<dataset-root>/aligned/train.jsonl
<dataset-root>/aligned/val.jsonl
<dataset-root>/aligned/test.jsonl
```

Each line must validate as `TrainingUtteranceArtifact`, and each row must contain:

- `audio_path`
- `canonical_phones`
- `phone_labels`
- `split`

`audio_path` may be absolute or relative to the dataset root.

These aligned files are expected to be produced from dataset-specific `prepared/*.jsonl` manifests first. See `docs/dataset_ingestion.md`.

### Step 7. Run actual feature precompute

Example for `libritts`:

```bash
export PRONUNCIATION_USE_HF_ENCODER=1
export PRONUNCIATION_DEVICE=cuda

python -m pronunciation_backend.training.precompute_features \
  --dataset libritts \
  --dataset-root /cold/pronunciation/datasets/libritts \
  --splits train val test \
  --backbone-id facebook/hubert-base-ls960 \
  --backbone-revision main \
  --embedding-source hubert \
  --alignment-source mfa \
  --pooling-version phone_mean_v1 \
  --artifact-schema-version phone_embedding_artifact_v1 \
  --sample-rate 16000 \
  --device cuda \
  --progress-every 100 \
  --shard-size 2000
```

This command auto-creates missing feature-store directories and manifests for the matching cache key before extraction starts.

For a smaller smoke test:

```bash
python -m pronunciation_backend.training.precompute_features \
  --dataset libritts \
  --dataset-root /cold/pronunciation/datasets/libritts \
  --splits train \
  --backbone-id facebook/hubert-base-ls960 \
  --backbone-revision main \
  --embedding-source hubert \
  --alignment-source mfa \
  --pooling-version phone_mean_v1 \
  --artifact-schema-version phone_embedding_artifact_v1 \
  --sample-rate 16000 \
  --device cuda \
  --progress-every 10 \
  --max-utterances 32 \
  --overwrite
```

The command prints periodic progress with:

- processed utterances in the current split
- accumulated phone rows
- utterances per second
- ETA in seconds

### Step 8. Inspect completion state

After a successful run:

- `state.json` should have `status: complete`
- `split_counts` should contain row counts
- `utterance_counts` should contain processed utterance counts
- `splits/<split>/part-*.jsonl` should exist

## What The Extractor Does

The current runner:

1. verifies the hashed feature-store exists
2. reads aligned `TrainingUtteranceArtifact` JSONL files
3. loads each waveform from `audio_path`
4. runs the frozen encoder
5. pools phone spans into `PhoneEmbeddingArtifact` rows
6. writes sharded JSONL files into the hashed split directory
7. updates `state.json`

## What The Actual Extractor Still Does Not Do

The current version intentionally keeps the first runnable pipeline simple.

It does not yet:

- batch multiple utterances in one forward pass
- write parquet shards
- estimate dataset-specific duration priors
- skip already processed individual utterances inside a split
- distribute work across multiple processes

Those can be added after the first full cache is generated successfully.

Recommended current output pattern:

```text
/cold/pronunciation/features/speechocean762/<feature_key>/splits/train/part-0000.jsonl
/cold/pronunciation/features/speechocean762/<feature_key>/splits/val/part-0000.jsonl
/cold/pronunciation/features/speechocean762/<feature_key>/splits/test/part-0000.jsonl
```

## Definition Of Done For This Stage

This feature-precompute stage is complete when:

- dataset roots are validated
- hashed feature-store directories are created
- manifests are written
- sharded feature files are written under split directories
- `state.json` records per-split row counts
