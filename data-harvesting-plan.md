# Pronunciation Data Harvesting Plan

## Why This Exists

The backend pipeline plan is technically solid, but the real execution risk is data availability.
This note translates the backend design into a concrete data harvesting strategy so we do not get
blocked later by missing supervision, weak native reference data, or license issues.

## Main Conclusion

We can start now if we keep v1 narrow.

The most practical first bundle is:

- `speechocean762` for supervised pronunciation scoring
- `L2-ARCTIC` for explicit phone-error patterns and alignment sanity checks
- `LibriSpeech` or `LibriTTS` for native-reference duration and embedding priors
- `CMUdict` for canonical ARPAbet pronunciations
- `Montreal Forced Aligner` models and dictionaries for offline alignment generation

This is enough to build an MVP around:

- `en-US` target accent only
- one canonical pronunciation per supported word
- constrained alignment against known phone sequences
- phone-level scoring rather than open-ended ASR

## Dataset Availability By Pipeline Stage

### 1. Learner speech with phone-level supervision

#### Recommended

- `speechocean762`
  - Host: [Hugging Face](https://huggingface.co/datasets/mispeech/speechocean762)
  - Mirror: [Kaggle](https://www.kaggle.com/datasets/tthieu0901/speechocean762)
  - Why it matters:
    - 5,000 English utterances
    - sentence-, word-, and phoneme-level expert scores
    - phoneme accuracy values and mispronunciation annotations
    - public and easy to load with `datasets`
  - Best use in this project:
    - train the phone scoring head
    - calibrate `match_score`
    - evaluate correlation with human phone labels
  - Risk:
    - all speakers are Mandarin L1, so it is not enough for broad fairness claims

#### Strong secondary source

- `L2-ARCTIC`
  - Official: [PSI Lab](https://psi.engr.tamu.edu/l2-arctic-corpus/)
  - Hugging Face processed mirror: [KoelLabs/L2ArcticSpontaneousSplit](https://huggingface.co/datasets/KoelLabs/L2ArcticSpontaneousSplit)
  - Kaggle mirror: [L2 Arctic Data](https://www.kaggle.com/datasets/divyamagg/l2-arctic-data)
  - Why it matters:
    - 24 non-native English speakers across Arabic, Hindi, Korean, Mandarin, Spanish, and Vietnamese L1s
    - about one hour of read speech per speaker
    - forced-aligned word and phoneme transcriptions
    - manually annotated subset with phone substitutions, deletions, and additions
  - Best use in this project:
    - error-pattern enrichment
    - debugging alignment behavior
    - stress-testing detection beyond Mandarin-accented English
  - Main warning:
    - license is `CC BY-NC 4.0`, so treat it as research-grade and not a guaranteed commercial-safe dependency

### 2. Native or near-native reference speech

#### Best practical public sources

- `LibriSpeech`
  - Official: [OpenSLR SLR12](https://www.openslr.org/12)
  - Hugging Face mirror: [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr)
  - Why it matters:
    - about 1,000 hours of read English speech
    - public, stable, and easy to obtain
    - already segmented and aligned at the utterance level
  - Best use in this project:
    - native duration priors
    - reference embeddings
    - alignment calibration bootstrap

- `LibriTTS`
  - Official: [OpenSLR SLR60](https://www.openslr.org/60/)
  - Why it matters:
    - about 585 hours of read English speech at 24 kHz
    - cleaner TTS-oriented preparation than LibriSpeech
    - original and normalized text available
  - Best use in this project:
    - duration modeling
    - native reference audio pool
    - cleaner corpus for acoustic statistics

#### Caveat for both

- These are very useful for bootstrapping native priors.
- They are not perfect if we want to make a strict claim like "General American canonical reference corpus."
- For MVP this is acceptable as long as we document the limitation.

### 3. Lexicon and canonical phone sequences

#### Required backbone

- `CMUdict`
  - Official: [The CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
  - Why it matters:
    - over 134,000 North American English words
    - ARPAbet pronunciations with stress markers
    - ideal for mapping a supported vocabulary to canonical phone sequences
  - Best use in this project:
    - word-to-phone lookup
    - IPA conversion layer
    - syllable and stress enrichment where needed

### 4. Forced alignment bootstrap

#### Recommended operational baseline

- `Montreal Forced Aligner`
  - Models page: [MFA pretrained models](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/models/index.html)
  - Model repo: [MontrealCorpusTools/mfa-models](https://github.com/MontrealCorpusTools/mfa-models)
  - Why it matters:
    - downloadable acoustic models, dictionaries, and G2P models
    - mature path for generating offline alignments and calibration artifacts
  - Best use in this project:
    - dataset generation
    - calibration corpus alignment
    - sanity-checking custom runtime aligner output

#### Important note

- MFA should be used to bootstrap and validate the data pipeline.
- We should still prefer a lightweight custom constrained aligner for production inference once the system is stable.

## Datasets To Treat Carefully

### Useful but not ideal as core v1 dependencies

- `TIMIT`
  - Hugging Face preview: [kimetsu/Timit](https://huggingface.co/datasets/kimetsu/Timit)
  - Access caveat: [manual download issue](https://github.com/huggingface/datasets/issues/4169)
  - Why it matters:
    - phonetically rich American English read speech
    - still valuable for phoneme experiments
  - Why not make it mandatory for v1:
    - more acquisition friction than LibriSpeech or LibriTTS
    - good for focused phoneme experiments, not the easiest first dependency

### Low-confidence sources

- Several recent Kaggle pronunciation datasets look lightly documented or tabular rather than corpus-grade.
- Example: [English Speech Pronunciation Assessment Dataset](https://www.kaggle.com/datasets/colabsss/english-speech-pronunciation-assessment-dataset)
- These may still be useful for idea generation or weak supervision, but they should not define the MVP unless we verify:
  - provenance
  - collection method
  - whether audio is truly present
  - whether labels are human-made or synthetic

## Recommended V1 Harvesting Order

### Phase 1. Must-have

1. Harvest `CMUdict`.
2. Harvest `speechocean762`.
3. Set up `Montreal Forced Aligner` English dictionary and acoustic models.
4. Harvest one native-reference corpus:
   - start with `LibriTTS` if cleaner audio is preferred
   - start with `LibriSpeech` if scale and tooling familiarity matter more

### Phase 2. Strongly recommended

1. Harvest `L2-ARCTIC`.
2. Convert its annotations into the project's canonical phone-event format.
3. Compare alignment and error behavior between `speechocean762` and `L2-ARCTIC`.

### Phase 3. Optional after MVP

1. Add `TIMIT` for targeted phoneme modeling experiments.
2. Add more native corpora only if we need stronger `en-US` calibration claims.
3. Add lower-confidence Kaggle datasets only after provenance review.

## Practical MVP Dataset Stack

If we want the smallest realistic stack that still supports the pipeline, use:

| Purpose | Dataset / Resource | Notes |
| --- | --- | --- |
| Supervised phone scoring | `speechocean762` | Strongest public first choice |
| Cross-accent learner errors | `L2-ARCTIC` | Great research value, non-commercial license |
| Native reference priors | `LibriTTS` or `LibriSpeech` | Good enough for bootstrap |
| Canonical pronunciations | `CMUdict` | Core lexicon dependency |
| Offline alignment | `Montreal Forced Aligner` | Bootstrap and validate |

## Risks We Should Track Early

### 1. Commercial licensing risk

- `L2-ARCTIC` is `CC BY-NC 4.0`.
- If this project is intended for commercial deployment, we should avoid making `L2-ARCTIC` the irreplaceable core training corpus.

### 2. Native-reference mismatch risk

- `LibriSpeech` and `LibriTTS` are strong bootstrap sources, but not perfect "General American canonical reference" corpora.
- This is acceptable for MVP if the supported vocabulary stays curated and small.

### 3. Fairness and generalization risk

- `speechocean762` is excellent, but Mandarin L1 only.
- A model trained mainly on it could overfit to Mandarin-accented error patterns.
- `L2-ARCTIC` helps mitigate this during research, but does not fully solve fairness.

### 4. Dataset sprawl risk

- We should not collect every speech dataset we can find.
- The fastest way to stall is to normalize ten heterogeneous corpora before proving the core loop.

## Recommended Project Decision

Commit to this data policy for backend v1:

- Use `speechocean762` as the primary supervised pronunciation dataset.
- Use `L2-ARCTIC` as a secondary research corpus, not a hard commercial dependency.
- Use `LibriTTS` or `LibriSpeech` for native-reference priors and calibration bootstrap.
- Use `CMUdict` plus `MFA` to build the alignment and lexicon foundation.
- Keep the supported vocabulary curated and small until the data pipeline is proven end to end.

## Saved Search Artifacts

The raw search outputs used to prepare this note are saved in the repository root:

- `pronunciation-datasets-search.json`
- `native-english-reference-datasets.json`
- `alignment-and-lexicon-resources.json`
- `l2-arctic-availability.json`
- `american-corpora-availability.json`
- `additional-pronunciation-datasets.json`
- `official-native-corpora-pages.json`
