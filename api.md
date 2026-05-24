# Backend REST API Contract

This document specifies the backend REST contracts implemented by `src/pronunciation_backend/main.py`.

## Base

- Protocol: HTTP
- Payloads: JSON responses and multipart form uploads
- Backend version metadata (from app): `0.2.0`

## Endpoints

### `GET /health`

Health and model-runtime metadata.

#### Success response

- Status: `200 OK`
- Content-Type: `application/json`
- Body:

```json
{
  "status": "ok",
  "model_ready": true,
  "runtime_backend": "scorer_v2",
  "model_version": "v2",
  "backbone_id": "facebook/hubert-base-ls960",
  "device": "cpu"
}
```

#### Field contract

- `status`: string, currently `"ok"`
- `model_ready`: boolean, currently `true`
- `runtime_backend`: string
- `model_version`: string
- `backbone_id`: string
- `device`: string

---

### `POST /v1/pronunciation/score`

Scores uploaded audio against one target word.

#### Request

- Content-Type: `multipart/form-data`
- Form fields:
  - `word` (required, string): target word
  - `audio` (required, file): audio upload
  - `speaker_id` (optional, string): accepted but currently not used in scoring
  - `noTrim` (optional, boolean, default `false`): skip backend auto-trim when `true`

#### Success response

- Status: `200 OK`
- Content-Type: `application/json`
- Body type: `PronunciationAssessmentResponse`

```json
{
  "word": "thought",
  "accent_target": "en-US",
  "ipa": "θɔt",
  "overall_score": 81.25,
  "confidence": 0.91,
  "audio_quality": {
    "status": "ok",
    "snr_estimate": 24.0,
    "duration_ms": 700,
    "rms": 0.2,
    "clipping_ratio": 0.0,
    "silence_ratio": 0.1,
    "original_duration_ms": 700,
    "trim_start_ms": 0,
    "trim_end_ms": 700,
    "trim_applied": false
  },
  "phonemes": [
    {
      "phoneme": "TH",
      "start_ms": 0,
      "end_ms": 120,
      "expected_score": 62.5,
      "expected_human_score": 1.1,
      "omission_probability": 0.02,
      "confidence": 0.88,
      "alignment_confidence": 0.93,
      "predicted_class": "accented",
      "quality_class_probs": {
        "wrong_or_missed": 0.11,
        "accented": 0.72,
        "correct": 0.17
      }
    }
  ],
  "primary_issue": {
    "phoneme": "TH",
    "type": "accented",
    "message": "phoneme TH is the main candidate for correction"
  },
  "reference": {
    "ipa": "θɔt",
    "audio_id": "thought_en_us_01",
    "asset_path": "assets/reference_audio/thought_en_us_01.wav"
  },
  "model_info": {
    "runtime_backend": "scorer_v2",
    "model_version": "v2",
    "checkpoint_name": "fake.pt",
    "backbone_id": "facebook/hubert-base-ls960",
    "device": "cpu",
    "class_labels": [
      "wrong_or_missed",
      "accented",
      "correct"
    ]
  }
}
```

## Schema Contract

### `PronunciationAssessmentResponse`

- `word`: string
- `accent_target`: literal `"en-US"`
- `ipa`: string
- `overall_score`: number, range `0..100`
- `confidence`: number, range `0..1`
- `audio_quality`: `AudioQualityPayload`
- `phonemes`: array of `PronunciationPhonePayload`
- `primary_issue`: `PrimaryIssuePayload`
- `reference`: `ReferencePayload` or `null` when the lexicon entry has no reference audio curation
- `model_info`: `ModelInfoPayload`

### `AudioQualityPayload`

- `status`: enum `ok | low_confidence | rejected`
- `snr_estimate`: number, `>= 0`
- `duration_ms`: integer, `>= 0`
- `rms`: number, `>= 0`
- `clipping_ratio`: number, `0..1`
- `silence_ratio`: number, `0..1`
- `original_duration_ms`: integer, `>= 0`
- `trim_start_ms`: integer, `>= 0`
- `trim_end_ms`: integer, `>= 0`
- `trim_applied`: boolean

### `PronunciationPhonePayload`

- `phoneme`: string
- `start_ms`: integer, `>= 0`
- `end_ms`: integer, `>= 0`
- `expected_score`: number, `0..100`
- `expected_human_score`: number, `>= 0`
- `omission_probability`: number, `0..1`
- `confidence`: number, `0..1`
- `alignment_confidence`: number, `0..1`
- `predicted_class`: enum `wrong_or_missed | accented | correct`
- `quality_class_probs`: `QualityClassProbabilitiesPayload`

### `QualityClassProbabilitiesPayload`

- `wrong_or_missed`: number, `0..1`
- `accented`: number, `0..1`
- `correct`: number, `0..1`

### `PrimaryIssuePayload`

- `phoneme`: string
- `type`: string
- `message`: string

Current runtime issue types emitted by response mapping include:

- `possibly_missing`
- `wrong_or_missed`
- `accented`
- `low_confidence`
- `none`
- `no_signal`

### `ReferencePayload`

- `ipa`: string
- `audio_id`: string or `null`
- `asset_path`: string or `null`

When present, `reference` carries optional listen-and-compare metadata. Scoring does not depend on reference audio. Omit `reference_audio_id` in the lexicon to return `reference: null`. When `reference_audio_id` is set but the manifest entry or asset file is missing, `asset_path` is `null`.

### `ModelInfoPayload`

- `runtime_backend`: string
- `model_version`: string
- `checkpoint_name`: string
- `backbone_id`: string
- `device`: string
- `class_labels`: array of strings

## Error Contract

### `400 Bad Request`

Audio validation failures (`AudioValidationError`), including:

- empty audio payload
- invalid/unsupported audio file
- audio too short
- audio too long

Response shape:

```json
{
  "detail": "Audio is too short for pronunciation scoring."
}
```

### `404 Not Found`

Unknown target word (`UnknownWordError`). The word token is empty, unsupported, or
not found in CMUdict.

Response shape:

```json
{
  "detail": "Word 'example' was not found in CMUdict."
}
```

### `503 Service Unavailable`

Alignment failures (`AlignmentError` family), including unavailable/misconfigured MFA, timeout, execution error, or unusable alignment output.

Response shape:

```json
{
  "detail": "MFA aligner is not configured"
}
```

### `500 Internal Server Error`

Unhandled runtime/config/file errors currently mapped in handler:

- `FileNotFoundError`
- `ValueError`

Response shape:

```json
{
  "detail": "Configured scorer checkpoint does not exist: ..."
}
```

### `422 Unprocessable Entity`

FastAPI validation errors for malformed/missing request fields (for example missing `word`, missing `audio`, invalid boolean for `noTrim`).

Response shape follows FastAPI default:

```json
{
  "detail": [
    {
      "loc": ["body", "..."],
      "msg": "...",
      "type": "..."
    }
  ]
}
```

## Behavioral Notes

- `word` lookup normalizes with `strip().lower()`.
- `speaker_id` is currently ignored (`reserved for future personalization`).
- By default, backend auto-trim is enabled before encoding/alignment/scoring.
- If `noTrim=true`, backend scores uploaded audio as-is (no auto-trim).
- Phoneme timestamps in response are relative to original uploaded timeline (trim offset reapplied).

