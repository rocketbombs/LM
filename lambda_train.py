#!/usr/bin/env python3
#
#Lambda Calculus β-Reduction Span Predictor Training Script
#===========================================================
#
#Production-grade training of a ~700M parameter model for predicting redex spans
#in λ-calculus terms under Levy-like reduction strategies.
#
#Architecture: Encoder-only Transformer with dual pointer heads optimized for
#high-SNR span prediction on structured symbolic sequences.
#
#Design rationale:
#  - Encoder-only selected over encoder-decoder: λ-terms are already fully
#    rendered; no generation needed, only pointer selection over fixed input.
#  - Right-sized architecture (18 layers × 1536 dim) provides ~700M params,
#    optimal for 16GB VRAM with gradient checkpointing and 8-bit Adam.
#  - RoPE positional encoding provides strong contextual representations for
#    nested parenthetical structure in λ-calculus terms.
#  - Dual pointer mechanism (start/end) mirrors successful approaches in QA
#    but adapted for symbolic rather than natural language tokens.
#  - GLU feed-forwards and RMSNorm for training stability at scale.
#
#Memory optimization for 16GB VRAM:
#  - Gradient checkpointing trades 30% compute for 60% memory reduction
#  - 8-bit AdamW optimizer reduces state from 16 bytes/param to 2 bytes/param
#  - Dynamic batching to token budgets prevents OOM on long sequences
#  - BF16 mixed precision for numerical stability on reduction chains
#
#Target: ~700M params, fits training with batch-tokens=24576 on RTX 5080.
#

import argparse
import json
import math
import os
import signal
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from queue import Queue
from threading import Thread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Optional dependencies with graceful fallbacks
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

# ============================================================================
# Configuration & CLI
# ============================================================================

@dataclass
class TrainingConfig:
    #Complete training configuration.#
    # Data
    train: str = '-'
    val: Optional[str] = None
    out: str = 'runs/levy700m'
    max_len: int = 2048
    truncate: str = 'tail'  # head, tail, middle
    
    # Training
    epochs: int = 3
    batch_tokens: int = 24576
    lr: float = 3e-4
    warmup: int = 2000
    wd: float = 0.01
    optimizer: str = 'adam8bit'
    grad_clip: float = 1.0
    
    # Model architecture (right-sized for ~700M params)
    d_model: int = 1536
    n_layers: int = 18
    n_heads: int = 12
    d_ff: int = 6144  # 4x for GLU
    dropout: float = 0.0
    drop_path: float = 0.1
    pos_encoding: str = 'rope'
    norm_type: str = 'rmsnorm'  # rmsnorm, layernorm
    
    # Loss weights
    alpha_ce: float = 0.8
    alpha_iou: float = 0.2
    label_smoothing: float = 0.05
    iou_window: int = 2
    nf_weight: float = 0.2  # Weight for normal form classification loss
    
    # Hardware
    amp: str = 'bf16'  # bf16, fp16, off
    grad_checkpoint: bool = False
    compile: bool = False
    flash: bool = True
    
    # Logging & checkpointing
    seed: int = 42
    save_interval: int = 1000
    eval_interval: int = 1000
    log_interval: int = 10
    tb: bool = False
    resume: Optional[str] = None
    dry_run: bool = False
    
    # Streaming
    stream_buffer: int = 10000
    val_samples: int = 1000


def parse_args() -> TrainingConfig:
    #Parse command-line arguments.#
    parser = argparse.ArgumentParser(
        description='Train λ-calculus β-reduction span predictor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--train', default='-', help='Training JSONL path or - for stdin')
    parser.add_argument('--val', help='Validation JSONL path')
    parser.add_argument('--out', default='runs/levy700m', help='Output directory')
    parser.add_argument('--max-len', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--truncate', choices=['head', 'tail', 'middle'], default='tail')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-tokens', type=int, default=24576, help='Token budget per step')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=int, default=2000, help='Warmup steps')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--optimizer', choices=['adamw', 'adam8bit'], default='adam8bit')
    parser.add_argument('--grad-clip', type=float, default=1.0)
    
    # Model
    parser.add_argument('--d-model', type=int, default=1536)
    parser.add_argument('--n-layers', type=int, default=18)
    parser.add_argument('--n-heads', type=int, default=12)
    parser.add_argument('--norm-type', choices=['rmsnorm', 'layernorm'], default='rmsnorm')
    
    # Hardware
    parser.add_argument('--amp', choices=['bf16', 'fp16', 'off'], default='bf16')
    parser.add_argument('--grad-checkpoint', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--flash', action='store_true', default=True)
    
    # Logging
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--tb', action='store_true', help='Enable TensorBoard')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


# ============================================================================
# Tokenizer with Offset Tracking
# ============================================================================

class LambdaTokenizer:
    #
    #Character-level tokenizer for λ-calculus with bijective offset mapping.
    #
    #Vocabulary: special tokens + ASCII printable chars optimized for λ-syntax.
    #Maintains token_idx → char_span mapping for span prediction accuracy.
    #
    
    def __init__(self):
        # Special tokens
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        
        # Core λ-calculus characters
        lambda_chars = ['\\', '.', '(', ')']
        digits = [str(i) for i in range(10)]
        letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        
        # Build vocabulary
        special = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab_list = special + lambda_chars + digits + letters + [' ']
        
        self.token2idx = {tok: idx for idx, tok in enumerate(vocab_list)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}
        self.vocab_size = len(vocab_list)
        
        # Special token IDs
        self.pad_id = self.token2idx[self.pad_token]
        self.bos_id = self.token2idx[self.bos_token]
        self.eos_id = self.token2idx[self.eos_token]
        self.unk_id = self.token2idx[self.unk_token]
    
    def encode(self, text: str, add_special: bool = True) -> Tuple[List[int], List[Tuple[int, int]]]:
        #
        #Encode text to token IDs with offset tracking.
        #
        token_ids = []
        offsets = []
        
        if add_special:
            token_ids.append(self.bos_id)
            offsets.append((-1, -1))  # Special token has no char span
        
        for i, char in enumerate(text):
            token_id = self.token2idx.get(char, self.unk_id)
            token_ids.append(token_id)
            offsets.append((i, i + 1))  # Char-level: each token is 1 char
        
        if add_special:
            token_ids.append(self.eos_id)
            offsets.append((-1, -1))
        
        return token_ids, offsets
    
    def decode(self, token_ids: List[int]) -> str:
        #Decode token IDs back to text.#
        chars = []
        for idx in token_ids:
            if idx in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            chars.append(self.idx2token.get(idx, self.unk_token))
        return ''.join(chars)
    
    def save(self, path: Path):
        #Save tokenizer config.#
        config = {
            'vocab_size': self.vocab_size,
            'token2idx': self.token2idx,
        }
        with open(path / 'tokenizer.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: Path):
        #Load tokenizer config.#
        with open(path / 'tokenizer.json', 'r') as f:
            config = json.load(f)
        self.token2idx = {k: int(v) for k, v in config['token2idx'].items()}
        self.idx2token = {int(v): k for k, v in self.token2idx.items()}
        self.vocab_size = config['vocab_size']


# ============================================================================
# Streaming Dataset
# ============================================================================

class TokenBudgetBatchSampler(torch.utils.data.Sampler):
    #
    #Batch sampler that groups examples by token budget with optimal packing.
    #
    #Pre-sorts dataset by length to minimize padding waste, then accumulates
    #sequences into batches where batch_size * max_length_in_batch ≤ token_budget.
    #This accurately models GPU memory consumption since sequences are padded to
    #the longest one in each batch.
    #
    #Uses bucketed shuffling to maintain length locality while preventing
    #curriculum effects from strict sorting.
    #
    
    def __init__(self, dataset: 'LambdaDataset', batch_tokens: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_tokens = batch_tokens
        self.shuffle = shuffle
        
        # Pre-compute lengths accounting for BOS/EOS tokens and max_len truncation
        self.lengths = [
            min(len(ex['term']) + 2, dataset.max_len) 
            for ex in dataset.examples
        ]
        self.indices = list(range(len(dataset)))
    
    def __iter__(self) -> Iterator[List[int]]:
        indices = list(self.indices)

        if self.shuffle:
            # Bucketed shuffle: shuffle within buckets to maintain length locality
            bucket_size = 128
            buckets = [indices[i:i+bucket_size] for i in range(0, len(indices), bucket_size)]
            for bucket in buckets:
                import random
                random.shuffle(bucket)
            indices = [idx for bucket in buckets for idx in bucket]

        mini_batch: List[int] = []
        max_len_in_batch = 0
        
        for idx in indices:
            seq_len = self.lengths[idx]
            
            # Update max length if adding this sample
            new_max_len = max(max_len_in_batch, seq_len)
            
            # Check if adding this sample would exceed budget
            # Actual memory footprint = num_samples * max_length (due to padding)
            if mini_batch and (len(mini_batch) + 1) * new_max_len > self.batch_tokens:
                # Yield current batch before adding this sample
                yield mini_batch
                mini_batch = []
                max_len_in_batch = 0
            
            # Add sample to current batch
            mini_batch.append(idx)
            max_len_in_batch = new_max_len
        
        # Yield final batch if non-empty
        if mini_batch:
            yield mini_batch
    
    def __len__(self) -> int:
        #Estimate number of batches.#
        total_tokens = sum(self.lengths)
        return max(1, total_tokens // self.batch_tokens)


class LambdaDataset(Dataset):
    #
    #Streaming JSONL dataset with length bucketing and dynamic batching.
    #
    #Loads examples into memory buffer, sorts by length for efficient packing,
    #and yields batches that fit within token budget.
    #
    
    def __init__(self, jsonl_path: str, tokenizer: LambdaTokenizer, 
                 max_len: int, buffer_size: int = 10000, truncate: str = 'tail'):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncate = truncate
        self.buffer_size = buffer_size

        self.examples: List[Dict[str, Any]] = []
        self._load_from_jsonl(jsonl_path, buffer_size)
    
    def _load_from_jsonl(self, path: str, max_examples: int):
        #Load examples from JSONL with optional reservoir sampling.#
        if path == '-':
            stream = sys.stdin
        else:
            stream = open(path, 'r')
        
        try:
            for i, line in enumerate(stream):
                if i >= max_examples:
                    break
                
                try:
                    example = json.loads(line.strip())
                    self.examples.append(example)
                except json.JSONDecodeError:
                    continue
        finally:
            if path != '-':
                stream.close()
        
        print(f"Loaded {len(self.examples)} examples from {path}")
        
        # Sort by term length for efficient batching
        print("Sorting examples by length for efficient batch packing...")
        self.examples.sort(key=lambda ex: len(ex['term']))
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        #
        #Process single example: tokenize, compute targets, handle truncation.
        #
        ex = self.examples[idx]
        
        # Tokenize term
        term = ex['term']
        token_ids, offsets = self.tokenizer.encode(term, add_special=True)
        
        # Handle truncation if needed
        if len(token_ids) > self.max_len:
            token_ids, offsets = self._truncate(token_ids, offsets, self.max_len)
        
        # Convert span from char offsets to token indices
        target_span = ex['target_span']
        start_char, end_char = target_span[0], target_span[1]
        
        # Find tokens that overlap with target span
        start_token_idx = 0
        end_token_idx = 0
        
        for i, (off_start, off_end) in enumerate(offsets):
            if off_start == -1:  # Special token
                continue
            if off_start <= start_char < off_end:
                start_token_idx = i
            if off_start < end_char <= off_end:
                end_token_idx = i
                break
        
        # Check if this is a normal form (no redex)
        is_nf = (start_char == 0 and end_char == 0)
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'offsets': offsets,
            'start_idx': start_token_idx,
            'end_idx': end_token_idx,
            'is_nf': is_nf,
            'steps_total': ex['steps_total'],
            'length': len(token_ids),
        }
    
    def _truncate(self, token_ids: List[int], offsets: List[Tuple[int, int]], 
                  max_len: int) -> Tuple[List[int], List[Tuple[int, int]]]:
        #Truncate sequence based on strategy.#
        if self.truncate == 'tail':
            # Keep BOS, truncate from end, add EOS
            return (token_ids[:max_len-1] + [self.tokenizer.eos_id],
                    offsets[:max_len-1] + [(-1, -1)])
        elif self.truncate == 'head':
            # Remove from start, keep EOS
            return ([self.tokenizer.bos_id] + token_ids[-(max_len-2):],
                    [(-1, -1)] + offsets[-(max_len-2):])
        else:  # middle
            # Keep BOS, EOS, remove middle
            half = (max_len - 2) // 2
            return ([token_ids[0]] + token_ids[1:half] + token_ids[-half:] + [token_ids[-1]],
                    [offsets[0]] + offsets[1:half] + offsets[-half:] + [offsets[-1]])


def collate_fn(batch: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    #
    #Collate batch with padding to max length in batch.
    #
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)
    
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    start_labels = torch.zeros(batch_size, dtype=torch.long)
    end_labels = torch.zeros(batch_size, dtype=torch.long)
    is_nf = torch.zeros(batch_size, dtype=torch.float)
    steps_total = torch.zeros(batch_size, dtype=torch.float)
    
    for i, item in enumerate(batch):
        seq_len = item['length']
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = True
        start_labels[i] = item['start_idx']
        end_labels[i] = item['end_idx']
        is_nf[i] = float(item['is_nf'])
        steps_total[i] = float(item['steps_total'])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_labels': start_labels,
        'end_labels': end_labels,
        'is_nf': is_nf,
        'steps_total': steps_total,
    }


# ============================================================================
# Model Components
# ============================================================================

class RMSNorm(nn.Module):
    #Root Mean Square Layer Normalization.#
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

        # Pre-compute positional encodings
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

        # Type hints for buffers (for IDE type checking)
        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.inv_freq: torch.Tensor
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    #Apply rotary embeddings to input tensor.#
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Rotate pairs
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    # Interleave back
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)


class GLUFeedForward(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, 
                 use_flash: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            q = apply_rotary_emb(q, rope_cos, rope_sin)
            k = apply_rotary_emb(k, rope_cos, rope_sin)
        
        # Reshape for attention: (B, n_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Build key padding mask: mask (B, L) has True=valid, invert for masking
        additive_mask = None
        if mask is not None:
            key_mask = ~mask  # True where padding
            # Create additive mask: large negative value for padded positions
            additive_mask = key_mask[:, None, None, :].to(q.dtype) * (-1e4)
        
        # Attention with Flash if available
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=additive_mask, 
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Standard attention
            scores = (q @ k.transpose(-2, -1)) * self.scale
            if additive_mask is not None:
                scores = scores + additive_mask
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = attn @ v
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    #Single transformer encoder block with GLU FFN.#
    
    def __init__(self, dim: int, n_heads: int, d_ff: int, dropout: float = 0.0,
                 drop_path: float = 0.0, norm_type: str = 'rmsnorm', use_flash: bool = True):
        super().__init__()
        
        NormLayer = RMSNorm if norm_type == 'rmsnorm' else nn.LayerNorm
        
        self.norm1 = NormLayer(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout, use_flash)
        self.norm2 = NormLayer(dim)
        self.ffn = GLUFeedForward(dim, d_ff, dropout)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                rope_cos: Optional[torch.Tensor] = None,
                rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Define checkpoint-friendly closures
        def _attn(x_):
            return self.attn(self.norm1(x_), mask, rope_cos, rope_sin)
        
        def _ffn(x_):
            return self.ffn(self.norm2(x_))
        
        # Apply with gradient checkpointing if enabled
        if self.training and hasattr(self, 'use_ckpt') and self.use_ckpt:
            x = x + self.drop_path(gradient_checkpoint(_attn, x, use_reentrant=False))
            x = x + self.drop_path(gradient_checkpoint(_ffn, x, use_reentrant=False))
        else:
            # Attention with residual
            x = x + self.drop_path(self.attn(self.norm1(x), mask, rope_cos, rope_sin))
            # FFN with residual
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class DropPath(nn.Module):
    #Stochastic depth (drop path) for regularization.#
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ============================================================================
# Main Model
# ============================================================================

class LambdaSpanPredictor(nn.Module):
    #
    #Encoder-only Transformer for λ-calculus redex span prediction.
    #
    #Architecture: 32-layer encoder (3072 dim, 24 heads) with dual pointer heads
    #for start/end span prediction. Total params ≈ 700M.
    #
    #Key design choices:
    #- RoPE positional encoding: Handles variable-length nested structures better
    #  than learned absolute positions; provides crucial relative position info
    #  for matching parentheses and lambda scopes.
    #- GLU feed-forwards: Gating mechanism improves gradient flow through deep
    #  network, critical for learning complex symbolic transformations.
    #- Dual pointer heads: Separate start/end prediction allows model to learn
    #  asymmetric span selection patterns in reduction strategies.
    #- Optional NF classifier: Binary head for detecting normal forms enables
    #  early termination and improves span prediction via joint training.
    #
    #Alternatives considered but not implemented:
    #- Encoder-decoder: Unnecessary complexity; terms already rendered, no
    #  generation needed. Pure encoder sufficient for fixed-input classification.
    #- Local+global attention mixing: Would reduce memory but λ-terms already
    #  relatively short (< 2K tokens) and reduction patterns require full context
    #  to identify outermost redexes. Full attention provides best SNR.
    #
    
    def __init__(self, config: TrainingConfig, vocab_size: int):
        super().__init__()
        self.config = config
        
        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        
        # Positional encoding
        self.rope: Optional[RotaryEmbedding]
        if config.pos_encoding == 'rope':
            self.rope = RotaryEmbedding(config.d_model // config.n_heads, config.max_len)
        else:
            self.rope = None
        
        # Transformer blocks with linearly increasing drop path
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.n_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, dpr[i], config.norm_type, config.flash
            ) for i in range(config.n_layers)
        ])
        
        # Final norm
        NormLayer = RMSNorm if config.norm_type == 'rmsnorm' else nn.LayerNorm
        self.norm = NormLayer(config.d_model)
        
        # Enable gradient checkpointing if requested
        if config.grad_checkpoint:
            for block in self.blocks:
                setattr(block, 'use_ckpt', True)
        
        # Prediction heads
        self.start_head = nn.Linear(config.d_model, 1)
        self.end_head = nn.Linear(config.d_model, 1)
        self.nf_head = nn.Linear(config.d_model, 1)  # Normal form classifier
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Validate RoPE max length
        if self.rope is not None:
            assert config.max_len <= self.rope.max_seq_len, \
                f"max_len ({config.max_len}) exceeds RoPE cache ({self.rope.max_seq_len})"
    
    def _init_weights(self, module):
        #Initialize weights with scaled normal distribution.#
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
                #Forward pass.
        B, L = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids)
        
        # Get RoPE embeddings if using
        rope_cos, rope_sin = None, None
        if self.rope is not None:
            rope_cos, rope_sin = self.rope(x, L)
            # Reshape for broadcasting: (L, head_dim) -> (1, 1, L, head_dim)
            rope_cos = rope_cos.unsqueeze(0).unsqueeze(0)
            rope_sin = rope_sin.unsqueeze(0).unsqueeze(0)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask, rope_cos, rope_sin)
        
        x = self.norm(x)
        
        # Pointer logits: (B, L)
        start_logits = self.start_head(x).squeeze(-1)
        end_logits = self.end_head(x).squeeze(-1)
        
        # Mask out padding positions
        if attention_mask is not None:
            start_logits = start_logits.masked_fill(~attention_mask, float('-inf'))
            end_logits = end_logits.masked_fill(~attention_mask, float('-inf'))
        
        # Normal form logit from pooled representation
        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        nf_logits = self.nf_head(pooled).squeeze(-1)
        
        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'nf_logits': nf_logits,
        }
    
    def count_parameters(self) -> int:
        #Count trainable parameters.#
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Loss Functions
# ============================================================================

def compute_span_loss(start_logits: torch.Tensor, end_logits: torch.Tensor,
                      start_labels: torch.Tensor, end_labels: torch.Tensor,
                      is_nf: torch.Tensor,
                      label_smoothing: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    #
    #Compute cross-entropy loss for start/end span prediction with label smoothing.
    #
    #For normal form examples (is_nf=1), the CE loss is masked out since there's
    #no valid redex to predict. The NF classifier head handles these cases.
    #
    # Create loss weights: zero out NF examples
    loss_weight = (1.0 - is_nf)  # (B,)
    
    # Cross-entropy with label smoothing
    start_loss_per_sample = F.cross_entropy(
        start_logits, start_labels, label_smoothing=label_smoothing, reduction='none'
    )
    end_loss_per_sample = F.cross_entropy(
        end_logits, end_labels, label_smoothing=label_smoothing, reduction='none'
    )
    
    # Apply weighting to mask NF examples
    start_loss = (start_loss_per_sample * loss_weight).sum() / (loss_weight.sum() + 1e-8)
    end_loss = (end_loss_per_sample * loss_weight).sum() / (loss_weight.sum() + 1e-8)
    
    return start_loss, end_loss


def compute_soft_iou_loss(start_logits: torch.Tensor, end_logits: torch.Tensor,
                          start_labels: torch.Tensor, end_labels: torch.Tensor,
                          window: int = 2) -> torch.Tensor:
    #
    #Compute soft IoU loss by creating triangular distributions around gold spans.
    #
    #This encourages the model to predict spans that overlap with the gold span,
    #providing a softer training signal than hard token-level CE.
    #
    B, L = start_logits.shape
    device = start_logits.device
    
    # Create soft targets: triangular distribution centered on gold
    def make_soft_target(labels: torch.Tensor) -> torch.Tensor:
        soft = torch.zeros(B, L, device=device)
        for i in range(B):
            center = labels[i].item()
            for offset in range(-window, window + 1):
                idx = center + offset
                if 0 <= idx < L:
                    weight = 1.0 - abs(offset) / (window + 1)
                    soft[i, idx] = weight
            # Normalize
            soft[i] = soft[i] / (soft[i].sum() + 1e-8)
        return soft
    
    start_soft = make_soft_target(start_labels)
    end_soft = make_soft_target(end_labels)
    
    # KL divergence between predicted distribution and soft target
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    
    start_iou_loss = F.kl_div(start_probs.log(), start_soft, reduction='batchmean')
    end_iou_loss = F.kl_div(end_probs.log(), end_soft, reduction='batchmean')
    
    return (start_iou_loss + end_iou_loss) / 2


def compute_nf_loss(nf_logits: torch.Tensor, is_nf: torch.Tensor) -> torch.Tensor:
    #Compute binary cross-entropy for normal form classification.#
    return F.binary_cross_entropy_with_logits(nf_logits, is_nf)


def compute_total_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                       config: TrainingConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    #
    #Compute combined loss with all components.
    #
    #Returns total loss and dictionary of individual loss components.
    #
    # Span losses (masked for NF examples)
    start_ce, end_ce = compute_span_loss(
        outputs['start_logits'], outputs['end_logits'],
        batch['start_labels'], batch['end_labels'],
        batch['is_nf'],
        config.label_smoothing
    )
    
    # Soft IoU loss
    iou_loss = compute_soft_iou_loss(
        outputs['start_logits'], outputs['end_logits'],
        batch['start_labels'], batch['end_labels'],
        config.iou_window
    )
    
    # Normal form loss (configurable weight for class imbalance)
    nf_loss = compute_nf_loss(outputs['nf_logits'], batch['is_nf'])
    
    # Combine with weights
    span_ce = (start_ce + end_ce) / 2
    total_loss = (
        config.alpha_ce * span_ce +
        config.alpha_iou * iou_loss +
        config.nf_weight * nf_loss
    )
    
    return total_loss, {
        'loss': total_loss.item(),
        'start_ce': start_ce.item(),
        'end_ce': end_ce.item(),
        'span_ce': span_ce.item(),
        'iou_loss': iou_loss.item(),
        'nf_loss': nf_loss.item(),
    }


# ============================================================================
# Metrics
# ============================================================================

def compute_span_metrics(outputs: Dict[str, torch.Tensor], 
                        batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    #
    #Compute span prediction metrics: exact match, token F1, span IoU.
    #
    start_preds = outputs['start_logits'].argmax(dim=-1)
    end_preds = outputs['end_logits'].argmax(dim=-1)
    
    start_labels = batch['start_labels']
    end_labels = batch['end_labels']
    
    # Exact match: both start and end correct
    exact_match = ((start_preds == start_labels) & (end_preds == end_labels)).float().mean()
    
    # Token-level F1
    start_correct = (start_preds == start_labels).float().mean()
    end_correct = (end_preds == end_labels).float().mean()
    token_f1 = (start_correct + end_correct) / 2
    
    # Span IoU
    def span_iou(pred_start, pred_end, gold_start, gold_end):
        #Compute IoU for single span pair.#
        intersection_start = max(pred_start, gold_start)
        intersection_end = min(pred_end, gold_end)
        intersection = max(0, intersection_end - intersection_start + 1)
        
        union_start = min(pred_start, gold_start)
        union_end = max(pred_end, gold_end)
        union = union_end - union_start + 1
        
        return intersection / (union + 1e-8)
    
    ious = []
    for i in range(len(start_preds)):
        iou = span_iou(
            start_preds[i].item(), end_preds[i].item(),
            start_labels[i].item(), end_labels[i].item()
        )
        ious.append(iou)
    
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    
    # NF accuracy
    nf_preds = (outputs['nf_logits'] > 0).float()
    nf_acc = (nf_preds == batch['is_nf']).float().mean()
    
    return {
        'exact_match': exact_match.item(),
        'token_f1': token_f1.item(),
        'span_iou': mean_iou,
        'nf_accuracy': nf_acc.item(),
    }


def highlight_span(text: str, start: int, end: int, marker: str = '«»') -> str:
    #
    #Highlight a character span in text with markers.
    
    
    if start == end == 0:
        return text + " [NF]"
    
    start_marker, end_marker = marker[0], marker[1]
    return text[:start] + start_marker + text[start:end] + end_marker + text[end:]


# ============================================================================
# Training Infrastructure
# ============================================================================

class Trainer:
    #
    #Main training orchestrator with checkpointing, logging, and evaluation.
    #
    
    def __init__(self, config: TrainingConfig, model: nn.Module, 
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 device: torch.device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup AMP
        self.scaler = None
        self.amp_dtype = None
        if config.amp != 'off':
            if config.amp == 'bf16' and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
                self.scaler = GradScaler()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.writer = None
        if config.tb:
            self.writer = SummaryWriter(config.out)

        # Metrics tracking
        self.train_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Setup output directory
        self.out_dir = Path(config.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Compile model if requested
        if config.compile:
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        #Create optimizer with optional 8-bit quantization.#
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer == 'adam8bit' and BITSANDBYTES_AVAILABLE:
            print("Using 8-bit AdamW optimizer")
            return bnb.optim.AdamW8bit(
                params, lr=self.config.lr, weight_decay=self.config.wd
            )
        else:
            if self.config.optimizer == 'adam8bit':
                print("Warning: bitsandbytes not available, falling back to AdamW")
            return torch.optim.AdamW(
                params, lr=self.config.lr, weight_decay=self.config.wd
            )
    
    def _create_scheduler(self):
        #Create learning rate scheduler with warmup and cosine annealing.#
        # Compute total steps based on current dataloader length
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = steps_per_epoch * self.config.epochs
        
        def lr_lambda(step):
            if step < self.config.warmup:
                # Linear warmup
                return step / max(1, self.config.warmup)
            # Cosine annealing after warmup
            progress = (step - self.config.warmup) / max(1, total_steps - self.config.warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _rebuild_scheduler(self):
        #Rebuild scheduler with updated step count (e.g., after OOM recovery).#
        steps_per_epoch = max(1, len(self.train_loader))
        total_steps = steps_per_epoch * self.config.epochs
        
        def lr_lambda(step):
            if step < self.config.warmup:
                return step / max(1, self.config.warmup)
            progress = (step - self.config.warmup) / max(1, total_steps - self.config.warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self):
        #Train for one epoch.#
        self.model.train()
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            step_start = time.time()
            
            # OOM auto-tuning on first batch
            if self.step == 0:
                success = self._try_train_step_with_oom_handling(batch)
                if not success:
                    raise RuntimeError("Failed to train even after reducing batch size. "
                                     "Try reducing --batch-tokens manually.")
            else:
                # Normal training step
                self._train_step(batch)
            
            step_time = time.time() - step_start
            self.train_metrics['step_time'].append(step_time)
            
            # Logging
            if self.step % self.config.log_interval == 0:
                self._log_training(batch_idx)
            
            # Evaluation
            if self.val_loader and self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                self._log_validation(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best')
            
            # Checkpointing
            if self.step % self.config.save_interval == 0:
                self.save_checkpoint(f'step_{self.step}')
            
            self.step += 1
            
            if self.config.dry_run and batch_idx >= 10:
                print(f"Dry run complete ({batch_idx} batches)")
                return
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {self.epoch} completed in {epoch_time:.1f}s")
    
    def _try_train_step_with_oom_handling(self, batch: Dict[str, Any]) -> bool:
        #
        #Try training step with automatic batch size reduction on OOM.
        #
        #Returns True if successful, False if failed after retries.
        #
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self._train_step(batch)
                if attempt > 0:
                    print(f"Successfully recovered with reduced batch_tokens={self.config.batch_tokens}")
                return True
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    if attempt < max_retries - 1:
                        # Reduce batch size by 20%
                        old_budget = self.config.batch_tokens
                        self.config.batch_tokens = int(self.config.batch_tokens * 0.8)
                        print(f"\nOOM detected. Reducing batch_tokens: {old_budget} → {self.config.batch_tokens}")
                        
                        # Clear cache
                        torch.cuda.empty_cache()
                        
                        # Rebuild dataloader with new budget
                        self._rebuild_dataloader()
                    else:
                        print(f"\nFailed after {max_retries} attempts to reduce batch size")
                        return False
                else:
                    raise
        
        return False
    
    def _rebuild_dataloader(self):
        #Rebuild train_loader with updated batch_tokens config.#
        dataset = self.train_loader.dataset
        tokenizer = dataset.tokenizer
        
        # Create new sampler with updated budget
        sampler = TokenBudgetBatchSampler(dataset, self.config.batch_tokens, shuffle=True)
        
        # Create new dataloader
        self.train_loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
            num_workers=0,
            pin_memory=True,
        )
        
        # Rebuild scheduler with updated step count
        self._rebuild_scheduler()
    
    def _train_step(self, batch: Dict[str, Any]):
        #Execute a single training step.#
        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with AMP
        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            outputs = self.model(batch['input_ids'], batch['attention_mask'])
            loss, loss_dict = compute_total_loss(outputs, batch, self.config)
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_span_metrics(outputs, batch)
        
        # Track metrics
        for k, v in loss_dict.items():
            self.train_metrics[k].append(v)
        for k, v in metrics.items():
            self.train_metrics[k].append(v)
    
    def evaluate(self) -> float:
        #Run evaluation on validation set.#
        self.model.eval()
        total_loss = 0
        all_metrics = defaultdict(list)
        qualitative_samples: List[str] = []

        if self.val_loader is None:
            return float('inf')

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    outputs = self.model(batch['input_ids'], batch['attention_mask'])
                    loss, loss_dict = compute_total_loss(outputs, batch, self.config)
                
                total_loss += loss.item()
                
                metrics = compute_span_metrics(outputs, batch)
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                
                # Collect qualitative samples (first 5 batches only)
                if batch_idx < 5:
                    self._collect_qualitative_samples(batch, outputs, qualitative_samples)
        
        self.model.train()
        
        # Log qualitative samples to TensorBoard
        if self.writer and qualitative_samples:
            self._log_qualitative_samples(qualitative_samples)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}
        
        return avg_loss
    
    def _collect_qualitative_samples(self, batch: Dict[str, torch.Tensor], 
                                     outputs: Dict[str, torch.Tensor],
                                     samples: List[str]):
        #Collect qualitative samples for TensorBoard logging.#
        if len(samples) >= 5:  # Limit to 5 samples
            return
        
        # Get predictions
        start_preds = outputs['start_logits'].argmax(dim=-1)
        end_preds = outputs['end_logits'].argmax(dim=-1)

        # Process first example in batch
        input_ids = batch['input_ids'][0].cpu().tolist()
        start_pred = int(start_preds[0].item())
        end_pred = int(end_preds[0].item())
        start_gold = int(batch['start_labels'][0].item())
        end_gold = int(batch['end_labels'][0].item())
        
        # Decode text (skip special tokens)
        text_tokens = [self.train_loader.dataset.tokenizer.idx2token.get(idx, '?') 
                      for idx in input_ids[1:-1]]  # Skip BOS/EOS
        text = ''.join(text_tokens)
        
        # Create highlighted versions
        # Subtract 1 only from start (to account for BOS), keep end as exclusive
        pred_text = highlight_span(text, start_pred - 1, end_pred, '⟨⟩')
        gold_text = highlight_span(text, start_gold - 1, end_gold, '⟨⟩')
        
        sample_text = f"Gold: {gold_text}\nPred: {pred_text}\n"
        samples.append(sample_text)
    
    def _log_qualitative_samples(self, samples: List[str]):
        #Log qualitative samples to TensorBoard.#
        if self.writer is None:
            return
        combined = "\n---\n".join(samples)
        self.writer.add_text('eval/samples', combined, self.step)
    
    def _log_training(self, batch_idx: int):
        #Log training metrics.#
        avg_metrics = {k: sum(v) / len(v) for k, v in self.train_metrics.items()}
        
        # Calculate actual throughput based on real batch size
        # Note: This is an approximation; actual padded tokens vary per batch
        tokens_per_sec = self.config.batch_tokens / avg_metrics.get('step_time', 1.0)
        
        # Console
        print(f"Step {self.step} | Loss: {avg_metrics['loss']:.4f} | "
              f"EM: {avg_metrics['exact_match']:.3f} | "
              f"IoU: {avg_metrics['span_iou']:.3f} | "
              f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
              f"{tokens_per_sec:.0f} tok/s")
        
        # TensorBoard
        if self.writer:
            for k, v in avg_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, self.step)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.step)
            self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, self.step)
    
    def _log_validation(self, val_loss: float):
        #Log validation metrics.#
        print(f"Validation loss: {val_loss:.4f}")
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.step)
    
    def save_checkpoint(self, name: str):
        #Save training checkpoint.#
        ckpt_path = self.out_dir / f'checkpoint_{name}.pt'
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
    
    def load_checkpoint(self, path: str):
        #Load training checkpoint.#
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from step {self.step}, epoch {self.epoch}")
    
    def train(self):
        #Main training loop.#
        print(f"\n{'='*60}")
        print("Starting training")
        print(f"{'='*60}\n")
        
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
                self.train_epoch()
                
                # Save epoch checkpoint
                self.save_checkpoint(f'epoch_{epoch}')
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.save_checkpoint('interrupted')
        
        finally:
            if self.writer:
                self.writer.close()
            print("\nTraining complete")


# ============================================================================
# Main Entry Point
# ============================================================================

def print_system_info(config: TrainingConfig, model: nn.Module):
    #Print system and model information.#
    print(f"\n{'='*60}")
    print("Lambda Calculus Span Predictor - Training Configuration")
    print(f"{'='*60}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Precision
    amp_mode = config.amp.upper() if config.amp != 'off' else 'FP32'
    print(f"Precision: {amp_mode}")
    
    # Flash attention
    flash_status = "Enabled" if config.flash and hasattr(F, 'scaled_dot_product_attention') else "Disabled"
    print(f"Flash Attention: {flash_status}")
    
    # Compilation
    compile_status = "Enabled" if config.compile else "Disabled"
    print(f"torch.compile: {compile_status}")
    
    # Gradient checkpointing
    gc_status = "Enabled" if config.grad_checkpoint else "Disabled"
    print(f"Gradient Checkpointing: {gc_status}")
    
    # Model parameters
    params = model.count_parameters()
    print(f"\nModel Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Training config
    print(f"\nBatch Token Budget: {config.batch_tokens:,}")
    print(f"Max Sequence Length: {config.max_len}")
    print(f"Learning Rate: {config.lr:.2e}")
    print(f"Weight Decay: {config.wd}")
    print(f"Epochs: {config.epochs}")
    
    print(f"\n{'='*60}\n")


def main():
    #Main entry point.#
    # Parse arguments
    config = parse_args()
    
    # Set seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create tokenizer
    print("Initializing tokenizer...")
    tokenizer = LambdaTokenizer()
    
    # Create datasets
    print(f"Loading training data from {config.train}...")
    train_dataset = LambdaDataset(
        config.train, tokenizer, config.max_len, 
        config.stream_buffer, config.truncate
    )
    
    val_dataset = None
    if config.val:
        print(f"Loading validation data from {config.val}...")
        val_dataset = LambdaDataset(
            config.val, tokenizer, config.max_len,
            config.val_samples, config.truncate
        )
    
    # Create data loaders with token budget batching
    print("Creating data loaders with token-budget batching...")
    
    # Create batch samplers that use dataset directly
    train_sampler = TokenBudgetBatchSampler(train_dataset, config.batch_tokens, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset:
        val_sampler = TokenBudgetBatchSampler(val_dataset, config.batch_tokens, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
            num_workers=0,
            pin_memory=True,
        )
    
    # Create model
    print("Initializing model...")
    model = LambdaSpanPredictor(config, tokenizer.vocab_size).to(device)
    
    # Print system info
    print_system_info(config, model)
    
    # Save configuration
    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    tokenizer.save(out_dir)
    
    # Create trainer
    trainer = Trainer(config, model, train_loader, val_loader, device)
    
    # Resume if requested
    if config.resume:
        print(f"Resuming from {config.resume}")
        trainer.load_checkpoint(config.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()