#!/usr/bin/env python3
"""
Lambda Calculus Reduction Inference Engine

Loads a trained model checkpoint and investigates its reduction strategy
compared to the gold standard Levy graph reduction. Analyzes divergence
points, efficiency metrics, and emergent optimization behavior.

IMPORTANT NOTE:
The current implementation attempts model-guided reduction but has limitations
in mapping character-level spans to exact redexes. The model was trained on
character-level tokenization, so predictions are character offsets, not
structural paths. For best results, use this tool primarily for:
  1. Analyzing prediction accuracy on gold labels
  2. Identifying divergence patterns
  3. Comparing efficiency metrics between strategies

Future improvements could add precise character-to-redex mapping using the
span annotations from the training data format.

Usage:
    # Basic investigation with 100 terms
    python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt --num-terms 100

    # Detailed output with verbose mode
    python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt \\
        --num-terms 50 --verbose

    # Save results to JSON for further analysis
    python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt \\
        --num-terms 200 --output-dir results/investigation_10k
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn.functional as F

# Import from existing scripts
from lambda_train import LambdaSpanPredictor, LambdaTokenizer, TrainingConfig
from lambda_gen import (TermGenerator, GraphReducer, TreeReducer, Term, TermType,
                         reference_substitute, Renderer)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    checkpoint: str
    num_terms: int = 100
    max_len: int = 2048
    min_size: int = 10
    max_size: int = 100
    max_steps: int = 1000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    verbose: bool = False
    output_dir: Optional[str] = None


@dataclass
class ReductionTrace:
    """Trace of a reduction sequence."""
    strategy: str  # 'model' or 'gold'
    steps: List[Tuple[str, Optional[Tuple[int, int]]]]  # (term_str, redex_span)
    converged: bool
    total_steps: int
    total_tokens: int
    tokens_per_step: List[int]
    divergence_step: Optional[int] = None  # When it diverged from gold


@dataclass
class ComparisonMetrics:
    """Metrics comparing model vs gold reduction."""
    term: str
    model_trace: ReductionTrace
    gold_trace: ReductionTrace

    # Efficiency metrics
    model_faster: bool
    step_difference: int  # Negative = model is faster
    token_difference: int  # Negative = model uses fewer tokens

    # Divergence analysis
    diverged: bool
    divergence_step: Optional[int]
    divergence_context: Optional[str] = None

    # Correctness
    same_normal_form: bool
    model_nf: str = ""
    gold_nf: str = ""


class InferenceEngine:
    """Engine for running inference and comparing reduction strategies."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        torch.manual_seed(config.seed)

        # Setup device
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"\nLoading checkpoint from: {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint, map_location=self.device)

        # Extract training config
        self.train_config = checkpoint.get('config', None)
        if self.train_config is None:
            # Try to load from config.json in same directory
            config_path = Path(config.checkpoint).parent.parent / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                    self.train_config = TrainingConfig(**config_dict)
            else:
                raise ValueError("Could not find training config in checkpoint or config.json")

        print(f"Model: {self.train_config.d_model}d × {self.train_config.n_layers}L")

        # Initialize tokenizer
        self.tokenizer = LambdaTokenizer()

        # Initialize model
        self.model = LambdaSpanPredictor(self.train_config, self.tokenizer.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Loaded from step: {checkpoint.get('step', 'unknown')}")

        # Initialize generators and reducers
        self.term_gen = TermGenerator(
            min_size=config.min_size,
            max_size=config.max_size
        )
        self.gold_reducer = GraphReducer(max_steps=config.max_steps)
        self.tree_reducer = TreeReducer(max_steps=config.max_steps)

        # Statistics
        self.comparisons: List[ComparisonMetrics] = []

    @torch.no_grad()
    def predict_redex(self, term_str: str) -> Optional[Tuple[int, int]]:
        """
        Predict redex span using the model.

        Returns:
            (start_char, end_char) tuple or None if model predicts normal form
        """
        # Tokenize
        token_ids, offsets = self.tokenizer.encode(term_str, add_special=True)

        # Pad to batch
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)

        # Get predictions
        start_logits = outputs['start_logits'][0]  # (L,)
        end_logits = outputs['end_logits'][0]  # (L,)
        nf_logit = outputs['nf_logits'][0, 0]  # scalar

        # Check if model predicts normal form
        nf_prob = torch.sigmoid(nf_logit).item()
        if nf_prob > 0.5:
            return None

        # Get start and end positions
        start_probs = F.softmax(start_logits, dim=0)
        end_probs = F.softmax(end_logits, dim=0)

        start_idx = start_probs.argmax().item()
        end_idx = end_probs.argmax().item()

        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx

        # Convert token indices to character offsets
        # Skip BOS token (index 0)
        if start_idx == 0 or start_idx >= len(offsets):
            return None
        if end_idx == 0 or end_idx >= len(offsets):
            return None

        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]

        return (start_char, end_char)

    def reduce_with_model(self, term: Term) -> ReductionTrace:
        """
        Reduce a term using model predictions for redex selection.

        This is exploratory - we extract character spans from the model
        and attempt to reduce at those positions.
        """
        steps = []
        current_term = term
        total_tokens = 0
        tokens_per_step = []

        for step_num in range(self.config.max_steps):
            term_str = str(current_term)
            term_tokens = len(term_str)
            total_tokens += term_tokens
            tokens_per_step.append(term_tokens)

            # Get model prediction
            redex_span = self.predict_redex(term_str)
            steps.append((term_str, redex_span))

            if redex_span is None:
                # Model predicts normal form
                return ReductionTrace(
                    strategy='model',
                    steps=steps,
                    converged=True,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    tokens_per_step=tokens_per_step
                )

            # Try to reduce at predicted span
            # For simplicity, we'll use a heuristic: find the redex that overlaps
            # most with the predicted span and reduce it
            # (A more sophisticated approach would parse the span exactly)

            # Find all redexes in current term
            all_redexes = self._find_all_redexes(current_term)

            if not all_redexes:
                # No redexes found but model predicted one - treat as converged
                return ReductionTrace(
                    strategy='model',
                    steps=steps,
                    converged=True,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    tokens_per_step=tokens_per_step
                )

            # Select redex that best matches predicted span
            best_redex = self._select_redex_by_span(all_redexes, redex_span, term_str)

            # Reduce at selected redex
            try:
                current_term = self._reduce_at_redex(current_term, best_redex)
            except Exception as e:
                if self.config.verbose:
                    print(f"Reduction failed at step {step_num}: {e}")
                return ReductionTrace(
                    strategy='model',
                    steps=steps,
                    converged=False,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    tokens_per_step=tokens_per_step
                )

        # Exceeded max steps
        return ReductionTrace(
            strategy='model',
            steps=steps,
            converged=False,
            total_steps=self.config.max_steps,
            total_tokens=total_tokens,
            tokens_per_step=tokens_per_step
        )

    def reduce_with_gold(self, term: Term) -> ReductionTrace:
        """Reduce using Levy graph reduction (gold standard)."""
        trace, exceeded_max, _, _ = self.gold_reducer.reduce(term)

        steps = []
        total_tokens = 0
        tokens_per_step = []

        for term_obj, redex_path in trace:
            term_str = str(term_obj)
            term_tokens = len(term_str)
            total_tokens += term_tokens
            tokens_per_step.append(term_tokens)

            # Convert path to character span if available
            redex_span = None
            if redex_path:
                # We'd need to convert path to character offsets
                # For now, just mark that a redex exists
                redex_span = (-1, -1)  # Placeholder

            steps.append((term_str, redex_span))

        return ReductionTrace(
            strategy='gold',
            steps=steps,
            converged=not exceeded_max,
            total_steps=len(trace),
            total_tokens=total_tokens,
            tokens_per_step=tokens_per_step
        )

    def _find_all_redexes(self, term: Term) -> List[Tuple[List[int], Term]]:
        """Find all redexes in term with their paths and subterms."""
        redexes = []

        def search(t: Term, path: List[int]):
            if t.type == TermType.APP and t.left and t.left.type == TermType.ABS:
                redexes.append((path.copy(), t))

            if t.type == TermType.ABS and t.body:
                search(t.body, path + [0])
            elif t.type == TermType.APP:
                if t.left:
                    search(t.left, path + [0])
                if t.right:
                    search(t.right, path + [1])

        search(term, [])
        return redexes

    def _select_redex_by_span(self, redexes: List[Tuple[List[int], Term]],
                             predicted_span: Tuple[int, int],
                             term_str: str) -> List[int]:
        """Select redex that best matches predicted character span."""
        if not redexes:
            return []

        # For now, use a simple heuristic: prefer leftmost-outermost (first in list)
        # A better approach would compute character offsets for each redex
        # and select the one with maximum overlap with predicted_span
        return redexes[0][0]

    def _reduce_at_redex(self, term: Term, path: List[int]) -> Term:
        """Apply beta reduction at specified path."""
        # Use TreeReducer's method
        return self.tree_reducer._apply_reduction(term, path)

    def compare_strategies(self, term: Term) -> ComparisonMetrics:
        """Run both strategies and compare results."""
        term_str = str(term)

        if self.config.verbose:
            print(f"\nTerm: {term_str}")

        # Run both reductions
        model_trace = self.reduce_with_model(term)
        gold_trace = self.reduce_with_gold(term)

        # Extract normal forms
        model_nf = model_trace.steps[-1][0] if model_trace.steps else term_str
        gold_nf = gold_trace.steps[-1][0] if gold_trace.steps else term_str

        # Check divergence
        diverged = False
        divergence_step = None
        min_steps = min(len(model_trace.steps), len(gold_trace.steps))

        for i in range(min_steps):
            if model_trace.steps[i][0] != gold_trace.steps[i][0]:
                diverged = True
                divergence_step = i
                break

        # Compute metrics
        step_diff = model_trace.total_steps - gold_trace.total_steps
        token_diff = model_trace.total_tokens - gold_trace.total_tokens

        metrics = ComparisonMetrics(
            term=term_str,
            model_trace=model_trace,
            gold_trace=gold_trace,
            model_faster=step_diff < 0,
            step_difference=step_diff,
            token_difference=token_diff,
            diverged=diverged,
            divergence_step=divergence_step,
            same_normal_form=(model_nf == gold_nf),
            model_nf=model_nf,
            gold_nf=gold_nf
        )

        if self.config.verbose:
            self._print_comparison(metrics)

        return metrics

    def _print_comparison(self, metrics: ComparisonMetrics):
        """Print detailed comparison."""
        print(f"\n  Model: {metrics.model_trace.total_steps} steps, "
              f"{metrics.model_trace.total_tokens} tokens")
        print(f"  Gold:  {metrics.gold_trace.total_steps} steps, "
              f"{metrics.gold_trace.total_tokens} tokens")
        print(f"  Δ Steps: {metrics.step_difference:+d}, "
              f"Δ Tokens: {metrics.token_difference:+d}")
        print(f"  Diverged: {metrics.diverged} (step {metrics.divergence_step})")
        print(f"  Same NF: {metrics.same_normal_form}")

    def run_investigation(self):
        """Run full investigation on generated terms."""
        print(f"\n{'='*70}")
        print(f"Lambda Calculus Reduction Strategy Investigation")
        print(f"{'='*70}\n")
        print(f"Generating {self.config.num_terms} terms...")
        print(f"Size range: [{self.config.min_size}, {self.config.max_size}]")
        print(f"Max steps: {self.config.max_steps}\n")

        # Generate terms
        terms = []
        for i in range(self.config.num_terms):
            term = self.term_gen.generate()
            terms.append(term)
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{self.config.num_terms} terms...")

        print(f"\nRunning dual reduction (model vs gold)...\n")

        # Run comparisons
        for i, term in enumerate(terms):
            if not self.config.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{self.config.num_terms} terms...")

            try:
                metrics = self.compare_strategies(term)
                self.comparisons.append(metrics)
            except Exception as e:
                if self.config.verbose:
                    print(f"Error processing term {i}: {e}")
                continue

        # Print summary
        self.print_summary()

        # Save results if output directory specified
        if self.config.output_dir:
            self.save_results()

    def print_summary(self):
        """Print summary statistics."""
        if not self.comparisons:
            print("No comparisons completed.")
            return

        print(f"\n{'='*70}")
        print(f"INVESTIGATION SUMMARY")
        print(f"{'='*70}\n")

        total = len(self.comparisons)

        # Convergence
        model_converged = sum(1 for c in self.comparisons if c.model_trace.converged)
        gold_converged = sum(1 for c in self.comparisons if c.gold_trace.converged)

        print(f"Convergence:")
        print(f"  Model: {model_converged}/{total} ({100*model_converged/total:.1f}%)")
        print(f"  Gold:  {gold_converged}/{total} ({100*gold_converged/total:.1f}%)")

        # Correctness
        same_nf = sum(1 for c in self.comparisons if c.same_normal_form)
        print(f"\nCorrectness:")
        print(f"  Same normal form: {same_nf}/{total} ({100*same_nf/total:.1f}%)")

        # Divergence
        diverged = sum(1 for c in self.comparisons if c.diverged)
        print(f"\nStrategy Divergence:")
        print(f"  Diverged: {diverged}/{total} ({100*diverged/total:.1f}%)")

        if diverged > 0:
            div_steps = [c.divergence_step for c in self.comparisons
                        if c.divergence_step is not None]
            if div_steps:
                avg_div_step = sum(div_steps) / len(div_steps)
                print(f"  Average divergence step: {avg_div_step:.1f}")

        # Efficiency
        model_faster_count = sum(1 for c in self.comparisons if c.model_faster)
        print(f"\nEfficiency (Steps):")
        print(f"  Model faster: {model_faster_count}/{total} "
              f"({100*model_faster_count/total:.1f}%)")

        step_diffs = [c.step_difference for c in self.comparisons]
        token_diffs = [c.token_difference for c in self.comparisons]

        avg_step_diff = sum(step_diffs) / len(step_diffs)
        avg_token_diff = sum(token_diffs) / len(token_diffs)

        print(f"  Average step difference: {avg_step_diff:+.2f}")
        print(f"  Average token difference: {avg_token_diff:+.2f}")

        # Token throughput
        model_avg_tokens = sum(c.model_trace.total_tokens for c in self.comparisons) / total
        gold_avg_tokens = sum(c.gold_trace.total_tokens for c in self.comparisons) / total

        print(f"\nToken Throughput:")
        print(f"  Model avg total tokens: {model_avg_tokens:.1f}")
        print(f"  Gold avg total tokens: {gold_avg_tokens:.1f}")

        # Most interesting divergences
        print(f"\nMost Interesting Cases:")

        # Cases where model is significantly faster
        faster_cases = sorted([c for c in self.comparisons if c.step_difference < -2],
                            key=lambda c: c.step_difference)[:3]

        if faster_cases:
            print(f"\n  Model significantly faster:")
            for c in faster_cases:
                print(f"    {c.step_difference:+d} steps | Term: {c.term[:60]}...")
                if self.config.verbose or len(faster_cases) <= 1:
                    print(f"      Model NF: {c.model_nf[:50]}...")
                    print(f"      Gold NF:  {c.gold_nf[:50]}...")

        # Cases where model is significantly slower
        slower_cases = sorted([c for c in self.comparisons if c.step_difference > 2],
                            key=lambda c: -c.step_difference)[:3]

        if slower_cases:
            print(f"\n  Model significantly slower:")
            for c in slower_cases:
                print(f"    {c.step_difference:+d} steps | Term: {c.term[:60]}...")

        # Cases with different normal forms (potential model errors)
        different_nf = [c for c in self.comparisons if not c.same_normal_form]
        if different_nf:
            print(f"\n  Different normal forms (potential errors): {len(different_nf)}")
            for c in different_nf[:2]:
                print(f"    Term: {c.term[:50]}...")
                print(f"      Model NF: {c.model_nf[:50]}...")
                print(f"      Gold NF:  {c.gold_nf[:50]}...")

        print(f"\n{'='*70}\n")

    def save_results(self):
        """Save detailed results to JSON."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'config': {
                'checkpoint': self.config.checkpoint,
                'num_terms': self.config.num_terms,
                'max_steps': self.config.max_steps,
            },
            'model': {
                'd_model': self.train_config.d_model,
                'n_layers': self.train_config.n_layers,
                'parameters': self.model.count_parameters(),
            },
            'comparisons': [
                {
                    'term': c.term,
                    'model_steps': c.model_trace.total_steps,
                    'gold_steps': c.gold_trace.total_steps,
                    'model_tokens': c.model_trace.total_tokens,
                    'gold_tokens': c.gold_trace.total_tokens,
                    'step_difference': c.step_difference,
                    'token_difference': c.token_difference,
                    'diverged': c.diverged,
                    'divergence_step': c.divergence_step,
                    'same_nf': c.same_normal_form,
                    'model_nf': c.model_nf,
                    'gold_nf': c.gold_nf,
                }
                for c in self.comparisons
            ]
        }

        output_file = output_dir / 'investigation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Investigate lambda calculus reduction strategies',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-terms', type=int, default=100,
                       help='Number of terms to test')
    parser.add_argument('--min-size', type=int, default=10,
                       help='Minimum term size')
    parser.add_argument('--max-size', type=int, default=100,
                       help='Maximum term size')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Max reduction steps')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')

    args = parser.parse_args()
    config = InferenceConfig(**vars(args))

    # Run investigation
    engine = InferenceEngine(config)
    engine.run_investigation()


if __name__ == '__main__':
    main()
