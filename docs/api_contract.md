# Inference API Contract

## Endpoint

`POST /v1/pronunciation/score`

## Request

Multipart form data:

- `word`: required target word from the curated `en-US` lexicon
- `audio`: required mono recording file
- `speaker_id`: optional reserved field for future personalization
- `noTrim`: optional boolean flag; when `true`, backend auto-trim is skipped and the uploaded clip is scored as-is

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
    "silence_ratio": 0.16,
    "original_duration_ms": 1480,
    "trim_start_ms": 430,
    "trim_end_ms": 1050,
    "trim_applied": true
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

When the lexicon entry has no `reference_audio_id`, the response omits reference metadata:

```json
{
  "reference": null
}
```

When reference curation exists but the asset file is unavailable, `reference.asset_path` is `null`.

## Per-Phoneme Semantics

- `expected_score`: pronunciation score projected into the dataset-aligned 0-100 scale
- `expected_human_score`: same prediction projected into the original 0-2 class scale
- `predicted_class`: the most likely pronunciation class
- `quality_class_probs`: class probabilities for `wrong_or_missed`, `accented`, and `correct`
- `omission_probability`: separate omission head output after sigmoid
- `confidence`: API-facing confidence that blends model certainty with alignment confidence
- `alignment_confidence`: MFA-derived alignment confidence from the runtime aligner
- phone times remain relative to the original uploaded clip even when backend trimming is applied

## Trimming Semantics

- by default the backend runs a conservative pre-alignment speech detector over the uploaded clip before encoding
- detection uses short-time RMS energy with smoothing, finds candidate speech islands, expands and merges nearby islands, and keeps the outer bounds of the plausible utterance region
- the intent is to remove obvious leading and trailing dead air for speed while preserving weak phones such as short stop releases that might otherwise be clipped away
- the trimmed clip is then aligned with MFA using the known target word transcript before phoneme features are pooled for the scorer
- `audio_quality.duration_ms` is the duration that was actually scored after trimming
- `audio_quality.original_duration_ms` is the full uploaded clip duration before trimming
- `audio_quality.trim_start_ms` and `audio_quality.trim_end_ms` mark the scored window within the original clip
- `audio_quality.trim_applied=false` means the backend kept the original clip unchanged
- set `noTrim=true` when the client already trimmed the clip and wants to bypass backend auto-trim

`primary_issue.type` is derived from the worst-scoring phoneme and can be one of:

- `possibly_missing`
- `wrong_or_missed`
- `accented`
- `low_confidence`
- `none`

## Error Cases

- `404`: target word token is empty, unsupported, or not found in CMUdict
- `400`: audio is empty, invalid, too short, or too long
- `503`: MFA is unavailable, times out, or fails to produce a usable alignment for the request

## Runtime Setup

The scoring backend stays inside the project `uv` environment and launches MFA through an explicit external command so it does not compete with the app environment.

Example:

```bash
export PRONUNCIATION_MFA_COMMAND="/opt/micromamba/bin/micromamba run -n mfa mfa"
export PRONUNCIATION_MFA_ACOUSTIC_MODEL=english_us_arpa
```

`PRONUNCIATION_MFA_COMMAND` should resolve to the `mfa` executable or to a launcher that ends by invoking `mfa`. The backend appends `align` and the request-specific corpus, dictionary, model, and output arguments automatically.

## Confidence Policy

- low-quality recordings reduce confidence, but do not directly rewrite model scores
- the backend still returns phoneme detail when possible
- clearly unusable audio is rejected before scoring
