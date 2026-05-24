# Resource Strategy

The MVP uses a curated `en-US` resource layer instead of a full pronunciation dictionary dump.

## Lexicon

`src/pronunciation_backend/resources/en_us_words.json`

Each entry stores:

- normalized word text
- canonical ARPAbet phone sequence
- IPA transcription
- reference audio id (optional)
- optional syllable grouping
- optional stress pattern

This keeps the scorer strict and avoids ambiguous words with multiple acceptable American pronunciations in v1.

## Reference Audio Manifest

`src/pronunciation_backend/resources/reference_audio_manifest.json`

Each entry stores:

- `audio_id`
- `word`
- `accent_target`
- `asset_path`

The API returns optional `reference` metadata with `audio_id` and `asset_path` when reference audio is curated, so the client can render a listen-and-compare UI without extra lookup calls. Scoring does not depend on reference audio.

## Expansion Rules

When expanding the vocabulary:

1. Add only one canonical `en-US` pronunciation per word.
2. Avoid highly variable words until multi-pronunciation support exists.
3. Keep ARPAbet and IPA aligned to the same canonical form.
4. Attach one stable reference audio asset per lexical entry when listen-and-compare metadata is desired.

## Recommended Next Resource Sources

- CMUdict for phone inventory seeding
- curated native `en-US` recordings for reference assets
- native `en-US` read speech corpora for duration priors and calibration
