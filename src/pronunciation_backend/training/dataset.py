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
    def __init__(self, jsonl_paths: list[Path]):
        super().__init__()
        self.jsonl_paths = jsonl_paths
        
    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            paths_to_process = self.jsonl_paths
        else:
            # Simple file-level sharding across workers
            paths_to_process = [
                path for i, path in enumerate(self.jsonl_paths)
                if i % worker_info.num_workers == worker_info.id
            ]
            
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
                        yield self._build_tensor_dict(current_phonemes)
                        current_utterance_id = utterance_id
                        current_phonemes = [row]
                    else:
                        current_phonemes.append(row)
                        
                if current_phonemes:
                    yield self._build_tensor_dict(current_phonemes)

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
            acoustic_features.append(torch.tensor(feats, dtype=torch.float32))
            
            phoneme_ids.append(torch.tensor(get_phoneme_id(p["phoneme"]), dtype=torch.long))
            
            # Extract targets
            match_targets.append(torch.tensor(p["regression_target"], dtype=torch.float32))
            
            # For duration target, we can use 100 for LibriTTS or something derived from z-score
            # but for now we just mimic the regression_target (since LibriTTS is perfectly aligned)
            duration_targets.append(torch.tensor(p["regression_target"], dtype=torch.float32))
            
            # Presence target (1.0 = present, 0.0 = omitted)
            presence_targets.append(torch.tensor(1.0 - p["omission_target"], dtype=torch.float32))
            
        return {
            "acoustic_features": torch.stack(acoustic_features), # (seq_len, 771)
            "phoneme_ids": torch.stack(phoneme_ids),             # (seq_len,)
            "match_targets": torch.stack(match_targets),         # (seq_len,)
            "duration_targets": torch.stack(duration_targets),   # (seq_len,)
            "presence_targets": torch.stack(presence_targets),   # (seq_len,)
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
