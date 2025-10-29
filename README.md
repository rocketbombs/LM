# Lambda Calculus Neural Reduction

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Rust 1.70+](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade framework for training neural networks to perform lambda calculus β-reduction using runtime-aware training data generation and Levy-style optimal reduction strategies.

## Overview

This project explores **learned reduction strategies** for lambda calculus by training transformer models on high-quality reduction traces. The system generates synthetic training data using graph-based call-by-need reduction (inspired by Levy's optimal reduction), then trains encoder-only transformers to predict the next redex location given a lambda term.

### Key Features

- **Runtime-Aware Training Data**: Metadata includes wall-clock timing, step costs, and pathological case detection
- **Dual Implementation**: Python for research/training, Rust for production data generation (484x faster)
- **Levy-Style Graph Reduction**: Call-by-need semantics with thunk memoization and sharing metrics
- **Normal Form Guarantee**: Complete reduction traces with no early stopping
- **Production-Grade Training**: Memory-optimized for 16GB GPUs with gradient checkpointing and 8-bit AdamW
- **Comprehensive Metrics**: 19+ metadata fields per training example for runtime awareness

### Performance

| Implementation | Throughput | vs Python |
|----------------|-----------|-----------|
| Python (single-threaded) | ~86 examples/s | 1x baseline |
| Rust (1 worker) | 19,787 examples/s | **230x faster** |
| Rust (8 workers) | 41,533 examples/s | **484x faster** |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Data Generation                        │
│  ┌──────────────────┐         ┌──────────────────┐     │
│  │  Rust Generator  │────────▶│  Python Pipeline │     │
│  │   (41k ex/s)     │  JSONL  │   (optional)     │     │
│  └──────────────────┘         └──────────────────┘     │
│           │                             │                │
│           │        Training Data        │                │
│           └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 Model Training                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Encoder-Only Transformer (75M-700M params)      │  │
│  │  - RoPE positional encoding                      │  │
│  │  - Dual pointer heads (start/end span)           │  │
│  │  - Gradient checkpointing + 8-bit AdamW          │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Inference                             │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Learned Reduction Strategy                      │  │
│  │  - Predict next redex span                       │  │
│  │  - Apply beta reduction                          │  │
│  │  - Compare vs Levy optimal baseline              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

**Python Implementation:**
```bash
# Python 3.7+
pip install torch numpy  # For training
```

**Rust Implementation (Optional, for high-throughput generation):**
```bash
# Rust 1.70+
cd lambda_gen_rs
cargo build --release
```

### Generate Training Data

**Using Rust (Recommended for production):**
```bash
cd lambda_gen_rs
cargo run --release -- generate training_data.jsonl 100000 8 100
# Generates 100k examples with 8 workers, 100ms wall clock limit per term
```

**Using Python:**
```bash
python lambda_gen.py --count 10000 --output training_data.jsonl
```

### Train a Model

```bash
python lambda_train.py \
    --train-data training_data.jsonl \
    --output-dir runs/levy_75m \
    --d-model 768 \
    --n-layers 8 \
    --batch-tokens 16384 \
    --steps 100000 \
    --lr 3e-4
```

**Memory-constrained training (16GB GPU):**
```bash
python lambda_train.py \
    --train-data training_data.jsonl \
    --output-dir runs/levy_75m_lowmem \
    --d-model 512 \
    --n-layers 6 \
    --batch-tokens 8192 \
    --gradient-checkpointing \
    --use-8bit-adam
```

### Run Inference

```bash
python lambda_infer.py \
    --checkpoint runs/levy_75m/checkpoints/step_100000.pt \
    --num-terms 100 \
    --verbose
```

## Project Structure

```
LM/
├── README.md                    # This file
├── lambda_gen.py               # Python data generator (reference implementation)
├── lambda_train.py             # Model training script
├── lambda_infer.py             # Inference and evaluation
├── parallel_gen.py             # Parallel Python generation
│
├── lambda_gen_rs/              # Rust high-performance generator
│   ├── Cargo.toml              # Zero external dependencies (std-only)
│   ├── VALIDATION.md           # Implementation validation report
│   ├── src/
│   │   ├── main.rs             # CLI interface
│   │   ├── generator.rs        # Term generation with custom RNG
│   │   ├── reduction.rs        # Graph reduction with wall clock limiting
│   │   ├── schema.rs           # Training data schema (19 metadata fields)
│   │   ├── parallel.rs         # Lock-free parallel pipeline
│   │   ├── render.rs           # De Bruijn rendering with span tracking
│   │   └── term.rs             # Arena-allocated term representation
│   └── target/                 # Build artifacts (excluded from git)
│
├── docs/                       # Documentation and analysis reports
│   ├── NF_REDUCTION_ANALYSIS.md         # Normal Form reduction verification
│   ├── IMPLEMENTATION_NOTES.md          # Implementation details
│   ├── INVESTIGATION_SUMMARY.md         # Performance investigation
│   ├── FUEL_BUDGET_INVESTIGATION.md     # Wall clock limiting design
│   ├── FUEL_BUDGET_RECOMMENDATIONS.md   # Budget tuning guide
│   └── RUNTIME_AWARE_SUMMARY.md         # Runtime awareness design
│
└── tests/                      # Test suite
    ├── test_nf_reduction.py              # Normal Form reduction tests
    ├── test_fuel_metrics.py              # Wall clock limiting tests
    ├── throughput_test.py                # Performance benchmarks
    ├── aggressive_fuel_test.py           # Stress tests
    ├── optimized_throughput_test.py      # Optimization validation
    ├── quick_throughput_test.py          # Quick perf checks
    └── runtime_aware_throughput_test.py  # Runtime metadata validation
```

## Training Data Schema

Each training example is a single JSONL line with the following structure:

```json
{
  "strategy": "levy_like",
  "render": "debruijn",
  "term": "(\\.(\\.(1 0)))(\\.0)",
  "step_k": 0,
  "target_span": [0, 17],
  "steps_total": 2,
  "diverged": false,
  "meta": {
    "size": 8,
    "depth": 4,
    "libs": 2,
    "seed": 12345,
    "draw_index": 0,
    "uid": "12345678-0000-0001",
    "thunk_evals": 3,
    "thunk_hits": 1,
    "schema_version": "v2",
    "term_hash": "a1b2c3d4",
    "step_ms": 0.15,
    "avg_step_ms": 0.15,
    "total_time_ms": 0.30,
    "wall_clock_limit_ms": 100.0,
    "time_remaining_ms": 99.70,
    "time_consumed_ratio": 0.003,
    "is_pathological": false,
    "size_growth_rate": 1.0,
    "initial_size": 8
  }
}
```

### Metadata Fields (19 total)

**Term Properties:**
- `size`: Number of nodes in term
- `depth`: Maximum nesting depth
- `libs`: Number of lambda binders
- `initial_size`: Starting term size

**Generation Tracking:**
- `seed`: RNG seed for reproducibility
- `draw_index`: Example index from this seed
- `uid`: Unique identifier
- `term_hash`: Hash of serialized term

**Sharing Metrics (Levy reduction):**
- `thunk_evals`: Thunks evaluated for first time
- `thunk_hits`: Cache hits on memoized thunks

**Runtime Awareness (Wall Clock):**
- `step_ms`: Duration of current reduction step
- `avg_step_ms`: Average step duration so far
- `total_time_ms`: Total time elapsed
- `wall_clock_limit_ms`: Budget limit
- `time_remaining_ms`: Budget remaining
- `time_consumed_ratio`: Fraction of budget used

**Pathological Detection:**
- `is_pathological`: True if >80% budget used, >5ms avg steps, or >3x growth
- `size_growth_rate`: current_size / initial_size

**Version:**
- `schema_version`: "v2"

## Key Design Decisions

### 1. Wall Clock as Primary Limiter

Unlike traditional abstract "fuel" budgets, this system uses **actual wall-clock timing** as the primary limiter:

- Check happens **before** expensive operations
- Per-step timing tracked in `step_ms`
- `max_steps` kept as safety fallback only
- Model learns real computational costs

See `docs/FUEL_BUDGET_INVESTIGATION.md` for detailed rationale.

### 2. Complete Reduction to Normal Form

The reducer **always continues until Normal Form** (no early stopping):

- `redex_path` returns `None` when NF reached
- Training data includes final step with `target_span = (0, 0)`
- Model learns to predict `(0, 0)` when term is in NF
- Compatible with Levy-style optimal reduction goals

See `docs/NF_REDUCTION_ANALYSIS.md` for verification tests.

### 3. std-only Rust Implementation

The Rust generator uses **zero external dependencies**:

- Custom LCG RNG (no `rand_chacha`)
- Manual JSON serialization (no `serde`)
- `std::thread` parallelism (no `rayon`)
- `std::sync::mpsc` channels (no `crossbeam`)

Benefits: Maximum portability, no network access needed, identical performance.

See `lambda_gen_rs/VALIDATION.md` for complete validation.

### 4. Pathological Case Detection

Terms that exhibit problematic behavior are flagged:

```python
is_pathological = (
    time_consumed_ratio > 0.8 or      # >80% budget used
    avg_step_ms > 5.0 or               # Slow steps
    size_growth_rate > 3.0 or          # Size tripled
    current_size > 200                 # Very large term
)
```

This allows models to learn early termination strategies.

## Model Architecture

### Encoder-Only Transformer

```
Input: "(\\.(\\.(1 0)))(\\. 0)"
       ↓
    Tokenizer (character-level)
       ↓
    Embedding (d_model=768)
       ↓
    RoPE Positional Encoding
       ↓
    Transformer Layers × 8
    ├─ Multi-Head Self-Attention
    ├─ RMSNorm
    ├─ GLU Feed-Forward
    └─ RMSNorm
       ↓
    Dual Pointer Heads
    ├─ Start Span Head → P(start position)
    └─ End Span Head   → P(end position)
```

**Why Encoder-Only?**
- Lambda terms are fully rendered (no generation needed)
- Task is pointer selection over fixed input (like extractive QA)
- More memory-efficient than encoder-decoder
- Allows larger models on same GPU budget

### Parameter Scaling

| Model Size | d_model | n_layers | Params | VRAM (16-bit) | VRAM (8-bit Adam) |
|-----------|---------|----------|---------|---------------|-------------------|
| Small     | 512     | 6        | ~38M    | 4 GB          | 2 GB              |
| Medium    | 768     | 8        | ~75M    | 8 GB          | 4 GB              |
| Large     | 1024    | 12       | ~150M   | 16 GB         | 8 GB              |
| XL        | 1536    | 18       | ~700M   | 64 GB         | 32 GB             |

## Usage Examples

### Generate Diverse Training Set

```bash
# Generate 1M examples across different sizes/complexities
for depth in 5 10 15 20; do
    cargo run --release -- generate \
        data/depth_${depth}.jsonl \
        250000 \
        8 \
        100 \
        --max-depth $depth
done
cat data/depth_*.jsonl > data/train_1m.jsonl
```

### Train with Curriculum Learning

```bash
# Stage 1: Small terms (depth 5)
python lambda_train.py \
    --train-data data/depth_5.jsonl \
    --output-dir runs/curriculum/stage1 \
    --steps 50000

# Stage 2: Medium terms (depth 10), resume from stage 1
python lambda_train.py \
    --train-data data/depth_10.jsonl \
    --output-dir runs/curriculum/stage2 \
    --resume-from runs/curriculum/stage1/checkpoints/step_50000.pt \
    --steps 50000

# Stage 3: Large terms (depth 15)
python lambda_train.py \
    --train-data data/depth_15.jsonl \
    --output-dir runs/curriculum/stage3 \
    --resume-from runs/curriculum/stage2/checkpoints/step_50000.pt \
    --steps 50000
```

### Evaluate Learned Strategy

```bash
python lambda_infer.py \
    --checkpoint runs/curriculum/stage3/checkpoints/step_50000.pt \
    --num-terms 500 \
    --output-dir results/analysis \
    --verbose

# Analyze results
python -c "
import json
from pathlib import Path

results = [json.loads(line) for line in Path('results/analysis/results.jsonl').read_text().splitlines()]
optimal_rate = sum(1 for r in results if r['optimal']) / len(results)
avg_steps_model = sum(r['steps_model'] for r in results) / len(results)
avg_steps_levy = sum(r['steps_levy'] for r in results) / len(results)

print(f'Optimal rate: {optimal_rate:.2%}')
print(f'Avg steps (model): {avg_steps_model:.1f}')
print(f'Avg steps (Levy):  {avg_steps_levy:.1f}')
"
```

## Testing

### Python Tests

```bash
# Normal Form reduction verification
python tests/test_nf_reduction.py

# Wall clock limiting tests
python tests/test_fuel_metrics.py

# Performance benchmarks
python tests/throughput_test.py
```

### Rust Tests

```bash
cd lambda_gen_rs
cargo test                          # All tests
cargo test test_unlimited_generation_no_overflow  # Specific test
cargo test --release                # Optimized build
```

### Benchmarks

```bash
# Python baseline
python tests/quick_throughput_test.py

# Rust comparison
cd lambda_gen_rs
cargo run --release -- benchmark 10000 8
```

## Performance Optimization

### Data Generation Bottlenecks

**Problem:** Python GIL prevents true parallelism (negative scaling with threads)

**Solution:** Rust implementation with:
- True parallelism via `std::thread`
- Per-worker RNG (no contention)
- Lock-free channels (`std::sync::mpsc`)
- Arena allocation for terms

**Result:** 484x speedup (8 workers)

### Training Memory Optimization

**Problem:** 700M parameter models don't fit on 16GB GPUs

**Solutions:**
1. **Gradient Checkpointing**: 60% memory reduction, 30% compute overhead
2. **8-bit AdamW**: Optimizer state from 16 bytes/param → 2 bytes/param
3. **Dynamic Batching**: Pack to token budget (e.g., 16384 tokens/batch)
4. **BF16 Mixed Precision**: Numerical stability without FP16 overflow

**Result:** 700M params trainable on 16GB GPU

## Documentation

Comprehensive documentation available in `docs/`:

- **[NF_REDUCTION_ANALYSIS.md](docs/NF_REDUCTION_ANALYSIS.md)**: Verification that reduction proceeds to Normal Form
- **[FUEL_BUDGET_INVESTIGATION.md](docs/FUEL_BUDGET_INVESTIGATION.md)**: Wall clock limiting design rationale
- **[IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md)**: Technical implementation details
- **[INVESTIGATION_SUMMARY.md](docs/INVESTIGATION_SUMMARY.md)**: Performance investigation and solutions
- **[RUNTIME_AWARE_SUMMARY.md](docs/RUNTIME_AWARE_SUMMARY.md)**: Runtime awareness design document
- **[lambda_gen_rs/VALIDATION.md](lambda_gen_rs/VALIDATION.md)**: Rust implementation validation

## Troubleshooting

### Issue: Low Training Throughput

**Symptoms:** <100 examples/sec training speed

**Diagnosis:**
```bash
# Check data loading bottleneck
python -c "
from lambda_train import LambdaDataset
import time
dataset = LambdaDataset('training_data.jsonl', max_len=2048)
start = time.time()
for i in range(1000):
    _ = dataset[i]
print(f'Load time: {time.time() - start:.2f}s for 1000 examples')
"
```

**Solutions:**
- Use Rust generator for faster data creation
- Increase `--num-workers` for data loading
- Reduce `--batch-tokens` if CPU-bound
- Use SSD for training data storage

### Issue: OOM During Training

**Symptoms:** CUDA out of memory errors

**Solutions:**
```bash
# Reduce batch size
--batch-tokens 8192

# Enable memory optimizations
--gradient-checkpointing --use-8bit-adam

# Reduce model size
--d-model 512 --n-layers 6

# Enable mixed precision
--amp  # BF16 if available
```

### Issue: Model Predicts (0,0) Always

**Symptoms:** Model learns to always predict Normal Form

**Diagnosis:** Class imbalance (most examples are intermediate steps)

**Solutions:**
- Weight loss by `1.0 / (step_k + 1)` to emphasize early steps
- Filter training data to include more non-NF examples
- Use focal loss to focus on hard examples

### Issue: Rust Compilation Fails

**Symptoms:** Cargo build errors

**Solutions:**
```bash
# Update Rust
rustup update

# Clean build
cargo clean
cargo build --release

# Check for std library issues
rustc --version  # Should be 1.70+
```

## Future Work

### Potential Enhancements

1. **Lamping Interaction Nets**: Implement true optimal reduction beyond Levy
2. **Distributed Generation**: Multi-machine data generation with coordination
3. **Learned Termination**: Model predicts when to stop (beyond NF detection)
4. **Meta-Learning**: Learn to adapt reduction strategy per term
5. **Compression**: JSONL compression for storage efficiency
6. **Schema Versioning**: Support multiple metadata schema versions
7. **Real-Time Validation**: Stream validation during generation
8. **GPU Acceleration**: CUDA kernels for reduction (research)

### Research Questions

- Can models learn optimal reduction orders beyond Levy?
- Do models discover novel reduction strategies?
- How does runtime awareness affect generalization?
- Can models learn to balance speed vs optimality?

## Citation

If you use this work in your research, please cite:

```bibtex
@software{lambda_neural_reduction,
  title = {Lambda Calculus Neural Reduction},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/LM}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`cargo test && python tests/test_nf_reduction.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Acknowledgments

- **Levy's Optimal Reduction**: Foundational work on graph-based lambda reduction
- **Lamping's Algorithm**: Optimal evaluation via interaction nets
- **Call-by-Need**: Lazy evaluation with memoization
- **Transformer Architecture**: Attention mechanisms for symbolic reasoning

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/LM/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/LM/discussions)

---

**Project Status**: Production-ready for data generation and training. Inference evaluation in progress.

**Last Updated**: October 2024
