# Resource Strategy

The runtime uses CMUdict for pronunciation lookup, with a small curated override
layer for reference audio and richer metadata.

## Runtime Vocabulary

Runtime scoring accepts words present in CMUdict. The backend loads CMUdict once at
startup from either:

- the `cmudict` Python package (default), or
- `PRONUNCIATION_CMUDICT_PATH` when a local dictionary file is configured.

When multiple CMUdict pronunciations exist for a word, the backend uses the first
listed variant deterministically.

## Curated Override Lexicon

`src/pronunciation_backend/resources/en_us_words.json`

Each curated entry stores:

- normalized word text
- canonical ARPAbet phone sequence
- IPA transcription
- reference audio id (optional)
- optional syllable grouping
- optional stress pattern

Curated entries override CMUdict for the same normalized word. This keeps starter
reference audio and richer MFA stress metadata available without limiting runtime
vocabulary to the curated set.

## Reference Audio Manifest

`src/pronunciation_backend/resources/reference_audio_manifest.json`

Each entry stores:

- `audio_id`
- `word`
- `accent_target`
- `asset_path`

The API returns optional `reference` metadata with `audio_id` and `asset_path`
when reference audio is curated, so the client can render a listen-and-compare UI
without extra lookup calls. Scoring does not depend on reference audio.

## Expansion Rules

When adding curated overrides:

1. Add only one canonical `en-US` pronunciation per word.
2. Avoid highly variable words until multi-pronunciation support exists.
3. Keep ARPAbet and IPA aligned to the same canonical form.
4. Attach one stable reference audio asset per lexical entry when listen-and-compare
   metadata is desired.

## Remaining Limitations

- Proper nouns, invented words, and non-English tokens may still return `404`.
- Most CMUdict-only words return `reference: null`.
- Multi-pronunciation scoring is not implemented yet.

## Recommended Next Resource Sources

- CMUdict for runtime phone inventory
- curated native `en-US` recordings for reference assets
- native `en-US` read speech corpora for duration priors and calibration
