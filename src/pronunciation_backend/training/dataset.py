import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence

from pronunciation_backend.training.cmudict_utils import ARPABET_TO_IPA, strip_phone_stress

# Build phoneme vocab mapping. 
# 0 = PAD, 1 = UNK
PHONEME_LIST = ["PAD", "UNK"] + sorted(list(ARPABET_TO_IPA.keys()))
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_LIST)}

def get_phoneme_id(phone: str) -> int:
    base = strip_phone_stress(phone)
    return PHONEME_TO_ID.get(base, 1)

class WordIterableDataset(IterableDataset):
    """
    Streams part-*.jsonl files and groups phonemes by utterance_id (word).
    """
    def __init__(self, jsonl_paths: list[Path], batch_size: int = 128, bucket_size_multiplier: int = 20):
        super().__init__()
        self.jsonl_paths = jsonl_paths
        self.batch_size = batch_size
        self.bucket_size_multiplier = bucket_size_multiplier
        
    def __iter__(self) -> Iterator[list[dict]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            paths_to_process = self.jsonl_paths
        else:
            # Simple file-level sharding across workers
            paths_to_process = [
                path for i, path in enumerate(self.jsonl_paths)
                if i % worker_info.num_workers == worker_info.id
            ]
            
        bucket = []
        for path in paths_to_process:
            with path.open('r', encoding='utf-8') as f:
                current_utterance_id = None
                current_phonemes = []
                
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    utterance_id = row["utterance_id"]
                    
                    if current_utterance_id is None:
                        current_utterance_id = utterance_id
                        
                    if utterance_id != current_utterance_id:
                        bucket.append(self._build_tensor_dict(current_phonemes))
                        current_utterance_id = utterance_id
                        current_phonemes = [row]
                        
                        if len(bucket) >= self.batch_size * self.bucket_size_multiplier:
                            # Sort bucket by seq_len to minimize padding overhead
                            bucket.sort(key=lambda x: x["seq_len"])
                            for i in range(0, len(bucket), self.batch_size):
                                yield bucket[i:i + self.batch_size]
                            bucket = []
                    else:
                        current_phonemes.append(row)
                        
                if current_phonemes:
                    bucket.append(self._build_tensor_dict(current_phonemes))
                    
        # Yield any remaining items in the bucket
        if bucket:
            bucket.sort(key=lambda x: x["seq_len"])
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def _build_tensor_dict(self, phonemes: list[dict]) -> dict:
        acoustic_features = []
        phoneme_ids = []
        match_targets = []
        duration_targets = []
        presence_targets = []
        
        for p in phonemes:
            # Concat acoustic features (771 dims total)
            # mean_embedding (768) + variance (1) + duration_z_score (1) + energy_mean (1)
            feats = p["mean_embedding"] + [
                p["variance"], 
                p["duration_z_score"], 
                p["energy_mean"]
            ]
            acoustic_features.append(feats)
            
            phoneme_ids.append(get_phoneme_id(p["phoneme"]))
            
            # Extract targets
            match_targets.append(p["regression_target"])
            
            # For duration target, we can use 100 for LibriTTS or something derived from z-score
            # but for now we just mimic the regression_target (since LibriTTS is perfectly aligned)
            duration_targets.append(p["regression_target"])
            
            # Presence target (1.0 = present, 0.0 = omitted)
            presence_targets.append(1.0 - p["omission_target"])
            
        return {
            "acoustic_features": torch.tensor(acoustic_features, dtype=torch.float32), # (seq_len, 771)
            "phoneme_ids": torch.tensor(phoneme_ids, dtype=torch.long),             # (seq_len,)
            "match_targets": torch.tensor(match_targets, dtype=torch.float32),         # (seq_len,)
            "duration_targets": torch.tensor(duration_targets, dtype=torch.float32),   # (seq_len,)
            "presence_targets": torch.tensor(presence_targets, dtype=torch.float32),   # (seq_len,)
            "seq_len": len(phonemes)
        }

def collate_word_batches(batch: list[dict]) -> dict:
    """
    Pads the variable length sequences in a batch to the max length in that batch.
    """
    # Extract lists of tensors
    acoustics = [item["acoustic_features"] for item in batch]
    p_ids = [item["phoneme_ids"] for item in batch]
    matches = [item["match_targets"] for item in batch]
    durations = [item["duration_targets"] for item in batch]
    presences = [item["presence_targets"] for item in batch]
    seq_lens = torch.tensor([item["seq_len"] for item in batch], dtype=torch.long)
    
    # Pad sequences
    # acoustic features padded with 0.0
    acoustics_padded = pad_sequence(acoustics, batch_first=True, padding_value=0.0)
    
    # phoneme ids padded with 0 (PAD token)
    p_ids_padded = pad_sequence(p_ids, batch_first=True, padding_value=0)
    
    # targets padded with 0.0
    matches_padded = pad_sequence(matches, batch_first=True, padding_value=0.0)
    durations_padded = pad_sequence(durations, batch_first=True, padding_value=0.0)
    presences_padded = pad_sequence(presences, batch_first=True, padding_value=0.0)
    
    # Create attention mask (True for valid positions, False for padded positions)
    batch_size = len(batch)
    max_len = acoustics_padded.size(1)
    attention_mask = torch.arange(max_len).expand(batch_size, max_len) < seq_lens.unsqueeze(1)
    
    return {
        "acoustic_features": acoustics_padded,
        "phoneme_ids": p_ids_padded,
        "match_targets": matches_padded,
        "duration_targets": durations_padded,
        "presence_targets": presences_padded,
        "attention_mask": attention_mask
    }
