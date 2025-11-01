# Lambda Calculus Inference Engine

High-performance Rust inference engine for comparing neural vs classical lambda calculus reduction.

## Features

- **Classical Normal-Order Reducer**: Pure leftmost-outermost β-reduction baseline
- **Neural Reducer**: ONNX model inference for learned reduction
- **Extreme Throughput**: Multi-threaded parallel reduction (default 16 workers)
- **Zero Python Overhead**: Pure Rust implementation with optimized inference

## Quick Start

### 1. Export PyTorch Model to ONNX

First, export your trained 4M model:

```python
import torch
from lambda_train import LambdaSpanPredictor, LambdaTokenizer, TrainingConfig

# Load your trained model
config = TrainingConfig(d_model=384, n_layers=4)
tokenizer = LambdaTokenizer()
model = LambdaSpanPredictor(config, tokenizer.vocab_size)

checkpoint = torch.load('runs/levy700m/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create dummy inputs for tracing
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 100))
dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.bool)

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    'model.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['start_logits', 'end_logits', 'nf_logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'seq_len'},
        'attention_mask': {0: 'batch', 1: 'seq_len'},
        'start_logits': {0: 'batch', 1: 'seq_len'},
        'end_logits': {0: 'batch', 1: 'seq_len'},
        'nf_logits': {0: 'batch'},
    },
    opset_version=14
)
print("Model exported to model.onnx")
```

### 2. Build the Inference Engine

```bash
cd lambda_gen_rs
cargo build --release --bin lambda-infer
```

### 3. Run Benchmarks

```bash
# Classical reducer only (baseline)
./target/release/lambda-infer \
  --classical-only \
  --terms 10000 \
  --workers 16 \
  --output results_classical.json

# Neural vs Classical comparison
./target/release/lambda-infer \
  --model ../model.onnx \
  --terms 10000 \
  --workers 16 \
  --output results_comparison.json
```

## Command Line Options

```
--model <PATH>          Path to ONNX model file
--terms <N>             Number of terms to evaluate (default: 1000)
--workers <N>           Number of parallel workers (default: 16)
--max-steps <N>         Maximum reduction steps per term (default: 1000)
--max-depth <N>         Maximum term depth for generation (default: 10)
--max-size <N>          Maximum term size for generation (default: 100)
--seed <N>              Random seed (default: 42)
--output <PATH>         Save JSON results to file
--classical-only        Run only classical reducer (skip neural)
```

## Example Output

```
Lambda Calculus Inference Engine
=================================
Workers: 16
Terms: 10000
Max steps: 1000

Generating 10000 test terms...
Generated 10000 terms

Running classical normal-order reducer...
  Total time: 1234.56ms
  Avg time/term: 0.1235ms
  Avg steps/term: 5.43
  Throughput: 8101 terms/s, 43988 steps/s
  Convergence: 100.0%

Running neural reducer...
  (Inference time: 234.56ms, 19.0% of total)
  Total time: 1234.56ms
  Avg time/term: 0.1235ms
  Avg steps/term: 3.21
  Throughput: 8101 terms/s, 26004 steps/s
  Convergence: 100.0%

=== COMPARISON ===
Speedup: 1.00x
Step efficiency: 59.15%
Neural uses 59.1% of classical steps
```

## Performance Expectations

With the 4M model achieving 99.95% exact match:

- **Classical Reducer**: ~8000-10000 terms/sec on 16 cores
- **Neural Reducer**: ~5000-8000 terms/sec (inference overhead)
- **Step Efficiency**: Neural should use ~60-80% of classical steps
- **Accuracy**: 99.95% exact match with classical normal form

## Architecture Notes

### Classical Reducer
- Pure tree-based β-reduction
- Leftmost-outermost strategy (normal order)
- No sharing or memoization
- Baseline for correctness and step count

### Neural Reducer
- ONNX model inference via `tract`
- Character-level tokenization
- Predicts next redex span + normal form classification
- Falls back to any-redex if span mapping fails

### Optimizations
- Parallel term reduction via `rayon`
- Zero-copy term representation
- Lock-free atomic counters
- Aggressive compiler optimizations (LTO, single codegen unit)

## Troubleshooting

**ONNX export fails**: Ensure you're using PyTorch 2.0+ and the model is in eval mode.

**Inference is slow**: Check that `tract-onnx` is using optimized BLAS. Consider using smaller batch sizes.

**Model predictions wrong**: Verify the ONNX model input/output names match the expected format.

## Next Steps

1. Export your trained 4M model to ONNX
2. Run classical baseline to verify correctness
3. Run neural comparison to measure speedup
4. Analyze step efficiency and accuracy

The goal is to demonstrate that your 4M neural model can match or exceed classical reduction performance while using fewer steps!
