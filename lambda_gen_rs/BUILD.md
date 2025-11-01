# Building the Inference Engine

## Prerequisites

The inference engine requires the following Rust crates:
- `tract-onnx` (0.21) - ONNX runtime
- `ndarray` (0.15) - Tensor operations
- `rayon` (1.8) - Parallel iteration
- `clap` (4.4) - CLI argument parsing
- `serde` + `serde_json` (1.0) - JSON serialization

## Setup

1. **Uncomment dependencies in `Cargo.toml`**:

```toml
[dependencies]
tract-onnx = "0.21"
ndarray = "0.15"
rayon = "1.8"
clap = { version = "4.4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

2. **Build the project**:

```bash
cd lambda_gen_rs
cargo build --release
```

3. **Run the classical reducer** (works without neural model):

```bash
./target/release/lambda-infer --classical-only --terms 10000
```

## File Structure

```
lambda_gen_rs/
├── src/
│   ├── classical.rs      # Classical normal-order reducer (ready to use)
│   ├── tokenizer.rs      # Character-level tokenizer (ready to use)
│   ├── neural.rs         # Neural inference (needs dependencies)
│   └── bin/
│       └── infer.rs      # Main benchmark binary (needs dependencies)
├── Cargo.toml            # Dependency configuration
├── BUILD.md              # This file
└── INFERENCE.md          # Usage instructions
```

## Status

- ✅ **Classical reducer**: Fully implemented, no external dependencies
- ✅ **Tokenizer**: Fully implemented, no external dependencies
- ⏸️ **Neural inference**: Requires `tract-onnx` and `ndarray`
- ⏸️ **Benchmark harness**: Requires `rayon`, `clap`, `serde`

The classical reducer can be tested independently once dependencies are enabled.

## Quick Test (Classical Only)

Once dependencies are uncommented and built, the classical reducer provides a fast baseline:

```bash
# Generate and reduce 10,000 terms with 16 workers
./target/release/lambda-infer \
  --classical-only \
  --terms 10000 \
  --workers 16 \
  --output results.json
```

Expected performance: ~8000-12000 terms/sec on modern hardware.
