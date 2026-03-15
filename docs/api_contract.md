# Inference API Contract

## Endpoint

`POST /v1/pronunciation/score`

## Request

Multipart form data:

- `word`: required target word from the curated `en-US` lexicon
- `audio`: required mono recording file
- `speaker_id`: optional reserved field for future personalization

## Response

```json
{
  "word": "thought",
  "accent_target": "en-US",
  "ipa": "θɔt",
  "overall_score": 74,
  "confidence": 0.88,
  "audio_quality": {
    "status": "ok",
    "snr_estimate": 21.4,
    "duration_ms": 620,
    "rms": 0.21,
    "clipping_ratio": 0.0,
    "silence_ratio": 0.16
  },
  "phonemes": [
    {
      "phoneme": "TH",
      "start_ms": 35,
      "end_ms": 150,
      "match_score": 49,
      "duration_score": 83,
      "presence_score": 95,
      "confidence": 0.91,
      "status": "low_match"
    }
  ],
  "primary_issue": {
    "phoneme": "TH",
    "type": "low_match",
    "message": "phoneme TH has the largest deviation"
  },
  "reference": {
    "ipa": "θɔt",
    "audio_id": "thought_en_us_01",
    "asset_path": "assets/reference_audio/thought_en_us_01.wav"
  }
}
```

## Status Semantics

Per-phoneme `status` is intentionally small:

- `ok`
- `low_match`
- `too_short`
- `too_long`
- `weak`
- `possibly_missing`
- `late_start`

## Error Cases

- `404`: target word is not in the curated MVP lexicon
- `400`: audio is empty, invalid, too short, or too long

## Confidence Policy

- low-quality recordings reduce overall confidence
- the backend still returns phoneme detail when possible
- clearly unusable audio is rejected before scoring
