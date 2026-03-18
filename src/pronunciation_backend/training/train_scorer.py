import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pronunciation_backend.training.scorer_model import PhonemeScorerModel
from pronunciation_backend.training.dataset import WordIterableDataset, collate_word_batches

def apply_negative_sampling(
    acoustic_features: torch.Tensor,
    phoneme_ids: torch.Tensor,
    match_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    attention_mask: torch.Tensor,
    prob: float = 0.15
):
    """
    Applies self-supervised negative sampling to simulate mispronunciations and omissions.
    Since LibriTTS has perfect native speech, we need to artificially inject errors
    so the model learns what "bad" sounds like.
    """
    batch_size, seq_len = phoneme_ids.size()
    device = phoneme_ids.device
    
    # Random floats for deciding augmentations
    rand_tensor = torch.rand(batch_size, seq_len, device=device)
    
    # 1. Substitution (prob/2): We tell the model to expect the WRONG phoneme.
    # It should learn that the acoustics don't match the expected phoneme, so match_score drops.
    sub_mask = (rand_tensor < (prob / 2)) & attention_mask
    random_phonemes = torch.randint(2, 42, (batch_size, seq_len), device=device) # 2-41 are valid phonemes
    phoneme_ids = torch.where(sub_mask, random_phonemes, phoneme_ids)
    match_targets = torch.where(sub_mask, torch.tensor(15.0, device=device), match_targets)
    
    # 2. Omission (prob/2): We zero out the acoustic features to simulate silence.
    # It should learn to output presence=0 and match_score=0.
    omit_mask = ((rand_tensor >= (prob / 2)) & (rand_tensor < prob)) & attention_mask
    acoustic_features = acoustic_features.clone()
    acoustic_features[omit_mask] = 0.0
    presence_targets = torch.where(omit_mask, torch.tensor(0.0, device=device), presence_targets)
    match_targets = torch.where(omit_mask, torch.tensor(0.0, device=device), match_targets)
    
    return acoustic_features, phoneme_ids, match_targets, presence_targets

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True, help="Path to the feature store split (e.g. /cold/.../splits/train)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", required=True, help="Where to save model weights")
    return parser

def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)
    
    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        raise FileNotFoundError(f"Features dir not found: {features_dir}")
        
    jsonl_paths = sorted(list(features_dir.glob("part-*.jsonl")))
    if not jsonl_paths:
        raise ValueError(f"No part-*.jsonl files found in {features_dir}")
        
    print(f"Found {len(jsonl_paths)} feature shard(s).")
    
    model = PhonemeScorerModel().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    # Loss functions
    # Using Smooth L1 (Huber) for regression is more robust to outliers than MSE
    reg_loss_fn = nn.SmoothL1Loss(reduction='none') 
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        dataset = WordIterableDataset(jsonl_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_word_batches,
            num_workers=2,
            pin_memory=True
        )
        
        model.train()
        epoch_match_loss = 0.0
        epoch_dur_loss = 0.0
        epoch_pres_loss = 0.0
        steps = 0
        
        start_time = time.time()
        
        for batch in dataloader:
            acoustics = batch["acoustic_features"].to(device)
            p_ids = batch["phoneme_ids"].to(device)
            matches = batch["match_targets"].to(device)
            durations = batch["duration_targets"].to(device)
            presences = batch["presence_targets"].to(device)
            mask = batch["attention_mask"].to(device)
            
            # Apply Self-Supervised Negative Sampling
            acoustics, p_ids, matches, presences = apply_negative_sampling(
                acoustics, p_ids, matches, presences, mask, prob=0.15
            )
            
            optimizer.zero_grad()
            
            outputs = model(
                acoustic_features=acoustics,
                phoneme_ids=p_ids,
                attention_mask=mask
            )
            
            # Calculate losses only on valid tokens (mask == True)
            # 1. Match Loss (Regression 0-100)
            m_loss = reg_loss_fn(outputs["match_score"], matches)
            m_loss = (m_loss * mask).sum() / max(1, mask.sum())
            
            # 2. Duration Loss (Regression 0-100)
            d_loss = reg_loss_fn(outputs["duration_score"], durations)
            d_loss = (d_loss * mask).sum() / max(1, mask.sum())
            
            # 3. Presence Loss (Binary Classification)
            p_loss = bce_loss_fn(outputs["presence_logit"], presences)
            p_loss = (p_loss * mask).sum() / max(1, mask.sum())
            
            # Weighted multi-task loss
            total_loss = m_loss + d_loss + (10.0 * p_loss) # Boost BCE weight since it's naturally smaller
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_match_loss += m_loss.item()
            epoch_dur_loss += d_loss.item()
            epoch_pres_loss += p_loss.item()
            steps += 1
            
            if steps % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"Step {steps:05d} | "
                    f"Match L: {m_loss.item():.2f} | "
                    f"Dur L: {d_loss.item():.2f} | "
                    f"Pres L: {p_loss.item():.4f} | "
                    f"{(args.batch_size * args.log_every) / elapsed:.1f} words/s"
                )
                start_time = time.time()
                
        # Save epoch checkpoint
        ckpt_path = checkpoint_dir / f"scorer_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")
        print(f"Epoch {epoch + 1} Averages -> Match: {epoch_match_loss/steps:.2f}, Dur: {epoch_dur_loss/steps:.2f}, Pres: {epoch_pres_loss/steps:.4f}")

if __name__ == "__main__":
    main()
