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
  "overall_score": 78.412,
  "confidence": 0.861,
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
      "expected_score": 48.917,
      "expected_human_score": 0.842,
      "omission_probability": 0.012441,
      "confidence": 0.812,
      "alignment_confidence": 0.93,
      "predicted_class": "accented",
      "quality_class_probs": {
        "wrong_or_missed": 0.181928,
        "accented": 0.617215,
        "correct": 0.200857
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
    "checkpoint_name": "scorer_v2_best.pt",
    "backbone_id": "facebook/hubert-base-ls960",
    "device": "cuda",
    "class_labels": ["wrong_or_missed", "accented", "correct"]
  }
}
```

## Per-Phoneme Semantics

- `expected_score`: pronunciation score projected into the dataset-aligned 0-100 scale
- `expected_human_score`: same prediction projected into the original 0-2 class scale
- `predicted_class`: the most likely pronunciation class
- `quality_class_probs`: class probabilities for `wrong_or_missed`, `accented`, and `correct`
- `omission_probability`: separate omission head output after sigmoid
- `confidence`: API-facing confidence that blends model certainty with alignment confidence
- `alignment_confidence`: heuristic alignment confidence from the runtime aligner

`primary_issue.type` is derived from the worst-scoring phoneme and can be one of:

- `possibly_missing`
- `wrong_or_missed`
- `accented`
- `low_confidence`
- `none`

## Error Cases

- `404`: target word is not in the curated MVP lexicon
- `400`: audio is empty, invalid, too short, or too long

## Confidence Policy

- low-quality recordings reduce confidence, but do not directly rewrite model scores
- the backend still returns phoneme detail when possible
- clearly unusable audio is rejected before scoring
