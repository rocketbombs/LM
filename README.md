# Lambda Calculus Neural Network Reduction

**Status:** Work in Progress - Experimental Research Project

## Overview

This project explores the application of neural networks to lambda calculus β-reduction. The goal is to train transformer models to predict reduction steps in lambda terms, potentially learning reduction strategies through pattern recognition.

### What is Lambda Calculus?

Lambda calculus is a formal system in mathematical logic for expressing computation based on function abstraction and application. β-reduction is the process of simplifying lambda terms by applying functions to their arguments.

### Project Purpose

This repository contains:
- **Training infrastructure** for teaching neural networks to predict lambda calculus reductions
- **Data generation tools** that create training examples from lambda term reductions
- **Inference systems** to compare learned strategies against classical reduction approaches
- **Evaluation utilities** for measuring model behavior

## Important Notes

- **No performance claims**: This is experimental research code. Any metrics or results should be considered preliminary and subject to change.
- **Work in progress**: The codebase is under active development and may contain incomplete features or experimental code.
- **Research quality**: This is research-grade code, not production software.

## Repository Structure

```
LM/
├── lambda_train.py              # Neural network training script
├── lambda_infer.py              # Inference and comparison engine
├── lambda_gen.py                # Python data generator (reference implementation)
├── parallel_gen.py              # Parallel data generation utilities
├── performance_metrics.py       # Metrics collection and analysis
├── export_onnx.py               # Model export utilities
├── diagnose_early_stopping.py   # Training diagnostics
│
├── lambda_gen_rs/               # High-performance Rust data generator
│   ├── src/                     # Rust source code
│   ├── BUILD.md                 # Build instructions
│   ├── VALIDATION.md            # Validation documentation
│   └── INFERENCE.md             # Inference notes
│
├── tests/                       # Test and diagnostic scripts
│   ├── diagnose_training_data.py
│   ├── check_diversity.py
│   └── ... (various test utilities)
│
└── docs/                        # Documentation
    ├── reference/               # Technical reference docs
    ├── issues/                  # Issue resolution history
    └── ... (various technical docs)
```

## Core Components

### 1. Data Generation

The system generates training data by:
- Creating random lambda terms with controlled complexity
- Reducing them using classical reduction strategies (normal-order, call-by-need)
- Recording the reduction sequence and redex positions
- Outputting structured training examples in JSONL format

Two implementations are available:
- **Python** (`lambda_gen.py`): Reference implementation, easier to modify
- **Rust** (`lambda_gen_rs/`): High-performance implementation for large-scale generation

### 2. Model Training

The training system (`lambda_train.py`) implements:
- Encoder-only transformer architecture
- Character-level tokenization of lambda terms
- Span prediction heads for identifying reduction positions
- Normal form detection
- Various optimization strategies (gradient checkpointing, mixed precision, etc.)

### 3. Inference and Evaluation

The inference engine (`lambda_infer.py`) can:
- Load trained models
- Run inference on lambda terms
- Compare neural predictions against classical reduction
- Generate evaluation metrics

## Getting Started

### Prerequisites

**For Python components:**
```bash
pip install torch numpy
```

**For Rust data generator (optional):**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the generator
cd lambda_gen_rs
cargo build --release
```

### Basic Usage

**Generate training data:**
```bash
# Using Python generator
python lambda_gen.py live --strategy normal --out train.jsonl --max-terms 10000

# Using Rust generator (faster)
cd lambda_gen_rs
cargo run --release -- generate train.jsonl 10000 8 250
```

**Train a model:**
```bash
python lambda_train.py \
    --train train.jsonl \
    --out runs/experiment_001 \
    --d-model 384 \
    --n-layers 4 \
    --batch-tokens 16384 \
    --epochs 10
```

**Run inference:**
```bash
python lambda_infer.py \
    --checkpoint runs/experiment_001/checkpoint_best.pt \
    --num-terms 100
```

## Architecture Details

### Lambda Term Representation

Terms are represented using de Bruijn indices:
- `\.body` - Lambda abstraction
- `f x` - Function application (left-associative)
- `0, 1, 2, ...` - Variable references (de Bruijn indices)

Example: The term `λx. λy. x y` becomes `\.\.1 0`

### Model Architecture

- **Type**: Encoder-only transformer
- **Tokenization**: Character-level on lambda term strings
- **Position encoding**: RoPE (Rotary Position Embeddings)
- **Prediction heads**:
  - Start/end span pointers for redex location
  - Normal form classifier
- **Training objective**: Cross-entropy + soft IoU loss for span prediction

### Reduction Strategies

The data generator supports multiple reduction strategies:
- **Normal order**: Leftmost-outermost reduction (tree-based)
- **Call-by-need**: Graph reduction with sharing and memoization

## Development and Testing

**Run validation suite:**
```bash
python lambda_gen.py validate --n 2000
```

**Check data diversity:**
```bash
python tests/check_diversity.py train.jsonl
```

**Diagnose training data:**
```bash
python tests/diagnose_training_data.py train.jsonl 10000
```

## Documentation

Additional documentation can be found in:
- `docs/` - Technical documentation and design notes
- `docs/issues/` - Resolved issues and their solutions
- `docs/reference/` - Reference material on specific topics
- `lambda_gen_rs/BUILD.md` - Rust build instructions
- `lambda_gen_rs/VALIDATION.md` - Data generator validation

## Research Context

This project explores several interesting questions:
- Can neural networks learn efficient reduction strategies?
- What patterns do models discover in lambda term structure?
- How do learned strategies compare to classical reduction orders?
- Can models generalize to terms larger than those seen during training?

## Limitations and Future Work

**Current limitations:**
- Model size and training time constrained by available compute
- Limited to de Bruijn representation (no named variables in training)
- Evaluation primarily on synthetic randomly-generated terms
- No theoretical guarantees on correctness or termination

**Potential future directions:**
- Explore different model architectures (decoder-only, encoder-decoder)
- Train on larger and more diverse term distributions
- Investigate learned strategies through model analysis
- Explore applications to program optimization or proof search

## Contributing

This is a research project and contributions are welcome, though the codebase is rapidly evolving. Please:
- Document any experimental features clearly
- Include tests for new functionality
- Update relevant documentation

## License

MIT License (or specify your license)

## Citation

If you use this code in your research, please cite this repository:

```bibtex
@software{lambda_neural_reduction_2024,
  title = {Lambda Calculus Neural Network Reduction},
  year = {2024},
  note = {Experimental research project}
}
```

---

**Disclaimer**: This is experimental research software under active development. Results, APIs, and implementations may change significantly.
