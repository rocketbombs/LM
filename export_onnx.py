#!/usr/bin/env python3
"""
Export trained PyTorch model to ONNX for Rust inference.

Usage:
    python export_onnx.py --checkpoint runs/levy700m/checkpoint_best.pt --output model.onnx
"""

import argparse
import torch
from pathlib import Path
from lambda_train import LambdaSpanPredictor, LambdaTokenizer, TrainingConfig


def export_to_onnx(checkpoint_path: str, output_path: str, max_seq_len: int = 2048):
    """Export PyTorch model to ONNX format."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        if isinstance(config_dict, dict):
            config = TrainingConfig(**config_dict)
        else:
            config = config_dict
    else:
        # Fallback to default config (adjust if needed)
        config = TrainingConfig(d_model=384, n_layers=4)
        print("Warning: Using default config")

    print(f"Model: {config.d_model}d × {config.n_layers}L")

    # Create model
    tokenizer = LambdaTokenizer()
    model = LambdaSpanPredictor(config, tokenizer.vocab_size)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")

    # Create dummy inputs
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 100), dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.bool)

    print(f"Exporting to ONNX: {output_path}")

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['start_logits', 'end_logits', 'nf_logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'seq_len'},
                'attention_mask': {0: 'batch', 1: 'seq_len'},
                'start_logits': {0: 'batch', 1: 'seq_len'},
                'end_logits': {0: 'batch', 1: 'seq_len'},
                'nf_logits': {0: 'batch'},
            },
            opset_version=14,
            do_constant_folding=True,
            verbose=False,
        )

    print(f"✓ Model exported successfully!")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Max sequence length: {max_seq_len}")
    print(f"  Output file: {output_path}")

    # Verify the export
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid!")

    # Print model info
    print("\nModel inputs:")
    for input in onnx_model.graph.input:
        print(f"  {input.name}: {input.type}")

    print("\nModel outputs:")
    for output in onnx_model.graph.output:
        print(f"  {output.name}: {output.type}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--checkpoint', required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--max-seq-len', type=int, default=2048, help='Maximum sequence length')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    try:
        export_to_onnx(args.checkpoint, args.output, args.max_seq_len)
        return 0
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
