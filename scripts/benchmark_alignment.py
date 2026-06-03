from __future__ import annotations

import io
import time
import logging
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
load_dotenv()

from pronunciation_backend.config import Settings
from pronunciation_backend.models import LexiconEntry
from pronunciation_backend.services.audio_prep import AudioPrepService
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.services.ctc_aligner import CtcForcedAligner
from pronunciation_backend.services.aligner import PhoneFeatureBuilder
from pronunciation_backend.services.response_mapper import ResponseMapper
from pronunciation_backend.services.scorer_runtime import ScorerModelInfo, ScorerPhonePrediction, ScorerRuntimeResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

class FakeScorerRuntime:
    def model_info(self) -> ScorerModelInfo:
        return ScorerModelInfo(
            runtime_backend="scorer_v2",
            model_version="v2",
            checkpoint_name="fake.pt",
            backbone_id="facebook/hubert-base-ls960",
            device="cpu",
            class_labels=("wrong_or_missed", "accented", "correct"),
        )

    def score(self, phone_features: list) -> ScorerRuntimeResult:
        return ScorerRuntimeResult(
            phone_predictions=[
                ScorerPhonePrediction(
                    phoneme=feature.phoneme,
                    start_ms=feature.start_ms,
                    end_ms=feature.end_ms,
                    expected_score=80.0,
                    expected_human_score=2.0,
                    omission_probability=0.01,
                    predicted_class="correct",
                    quality_class_probs={"wrong_or_missed": 0.05, "accented": 0.05, "correct": 0.90},
                    alignment_confidence=feature.alignment_confidence,
                )
                for feature in phone_features
            ],
            model_info=self.model_info(),
        )

def generate_dummy_wav_bytes(duration_seconds: float = 1.5, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    # 440Hz sine wave
    samples = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def main() -> None:
    print("=" * 60)
    print("PRONUNCIATION PIPELINE STAGE-TIMING BENCHMARK")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    settings = Settings(
        use_hf_encoder=True,
        backbone_id="facebook/hubert-base-ls960",
        device=device,
        aligner_backend="ctc",
        ctc_model_id="bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
        ctc_device=device,
    )
    
    # Instantiate real services
    print("Initializing services (this may download models if not cached)...")
    audio_prep = AudioPrepService(settings)
    feature_encoder = SSLFeatureEncoder(settings)
    ctc_aligner = CtcForcedAligner(settings.ctc_model_id, settings.ctc_device)
    feature_builder = PhoneFeatureBuilder()
    scorer_runtime = FakeScorerRuntime()
    response_mapper = ResponseMapper()
    
    # Warm up models
    print("Warming up models...")
    dummy_wav = generate_dummy_wav_bytes()
    entry = LexiconEntry(word="cat", phones=["K", "AE", "T"], ipa="kæt")
    
    prepared = audio_prep.decode(dummy_wav, enable_trim=True)
    encoded = feature_encoder.encode(prepared)
    _ = ctc_aligner.align(entry, prepared, encoded)
    print("Warmup complete.")
    
    # Benchmark runs
    num_runs = 10
    timings = {
        "audio_prep": [],
        "hubert_encode": [],
        "ctc_align": [],
        "feature_build": [],
        "scorer_v2": [],
        "response_map": [],
        "total_end_to_end": [],
    }
    
    for i in range(num_runs):
        t_start = time.perf_counter()
        
        t0 = time.perf_counter()
        prepared = audio_prep.decode(dummy_wav, enable_trim=True)
        t1 = time.perf_counter()
        timings["audio_prep"].append(t1 - t0)
        
        t0 = time.perf_counter()
        encoded = feature_encoder.encode(prepared)
        t1 = time.perf_counter()
        timings["hubert_encode"].append(t1 - t0)
        
        t0 = time.perf_counter()
        spans = ctc_aligner.align(entry, prepared, encoded)
        t1 = time.perf_counter()
        timings["ctc_align"].append(t1 - t0)
        
        t0 = time.perf_counter()
        phone_features = feature_builder.build(encoded, spans)
        t1 = time.perf_counter()
        timings["feature_build"].append(t1 - t0)
        
        t0 = time.perf_counter()
        runtime_result = scorer_runtime.score(phone_features)
        t1 = time.perf_counter()
        timings["scorer_v2"].append(t1 - t0)
        
        t0 = time.perf_counter()
        _ = response_mapper.build_response(
            word=entry.word,
            ipa=entry.ipa,
            prepared_audio=prepared,
            runtime_result=runtime_result,
            reference=None,
        )
        t1 = time.perf_counter()
        timings["response_map"].append(t1 - t0)
        
        t_end = time.perf_counter()
        timings["total_end_to_end"].append(t_end - t_start)
        
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS (Average over {num_runs} runs)")
    print("=" * 60)
    print(f"| {'Pipeline Stage':<25} | {'Average Latency (ms)':<20} |")
    print(f"| {'-'*25} | {'-'*20} |")
    for stage, times in timings.items():
        avg_ms = np.mean(times) * 1000.0
        print(f"| {stage:<25} | {avg_ms:>17.2f} ms |")
    print("=" * 60)

if __name__ == "__main__":
    main()
