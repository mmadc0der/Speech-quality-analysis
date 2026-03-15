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
- split list
- backbone id
- backbone revision
- adapter or LoRA id
- embedding source
- alignment source
- pooling version
- artifact schema version
- sample rate

This lets you safely keep multiple cached feature sets side by side. If you later introduce LoRA or a different pooling method, it automatically lands in a new directory rather than silently mixing artifacts.

## Current Planning Utility

The repository now includes a small planning utility:

`python -m pronunciation_backend.training.feature_store`

Supported commands:

- `plan`: create the hashed feature-store directory and manifests
- `verify`: check dataset presence and feature-store manifests

This utility does not yet extract embeddings. It is the control-plane layer for the future precompute runner.

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

### Step 3. Create a hashed feature-store plan for `speechocean762`

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

### Step 4. Verify the planned feature-store directory

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

### Step 5. Repeat the same plan for native-reference data

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

## What The Actual Extractor Must Do Next

The next implementation step should add a real precompute runner that:

1. loads aligned utterance artifacts for one dataset split
2. runs the frozen backbone offline
3. pools frame embeddings into `PhoneEmbeddingArtifact` rows
4. writes sharded parquet files into the hashed split directory
5. updates `state.json` with row counts and completion status

Recommended output pattern:

```text
/cold/pronunciation/features/speechocean762/<feature_key>/splits/train/part-0000.parquet
/cold/pronunciation/features/speechocean762/<feature_key>/splits/val/part-0000.parquet
/cold/pronunciation/features/speechocean762/<feature_key>/splits/test/part-0000.parquet
```

## Definition Of Done For This Stage

This feature-precompute stage is complete when:

- dataset roots are validated
- hashed feature-store directories are created
- manifests are written
- later extractor runs can detect whether a compatible cache already exists

The stage is not yet complete when only raw datasets are downloaded. The cache namespace and manifests must exist first.
