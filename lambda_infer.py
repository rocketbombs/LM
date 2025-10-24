#!/usr/bin/env python3
#
# Lambda Calculus Reduction Inference Engine
#
# Loads a trained model checkpoint and investigates its reduction strategy
# compared to the gold standard Levy graph reduction. Analyzes divergence
# points, efficiency metrics, and emergent optimization behavior.
#
# Usage:
#   # Basic investigation with 100 terms
#   python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt --num-terms 100
#
#   # Detailed output with verbose mode
#   python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt \
#       --num-terms 50 --verbose
#
#   # Save results to JSON for further analysis
#   python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt \
#       --num-terms 200 --output-dir results/investigation_10k

import argparse
import json
import random
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
                         reference_substitute, Renderer, RenderResult)


@dataclass
class InferenceConfig:
    # Configuration for inference engine.
    checkpoint: str
    num_terms: int = 100
    max_len: int = 2048
    min_depth: int = 2
    max_depth: int = 10
    max_size: int = 100
    max_steps: int = 1000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    verbose: bool = False
    output_dir: Optional[str] = None


@dataclass
class ReductionTrace:
    # Trace of a reduction sequence.
    strategy: str  # 'model' or 'gold'
    steps: List[Tuple[str, Optional[Tuple[int, int]]]]  # (term_str, redex_span)
    converged: bool
    total_steps: int
    total_tokens: int
    tokens_per_step: List[int]
    divergence_step: Optional[int] = None  # When it diverged from gold


@dataclass
class ComparisonMetrics:
    # Metrics comparing model vs gold reduction.
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
    # Engine for running inference and comparing reduction strategies.

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
        rng = random.Random(config.seed)
        self.term_gen = TermGenerator(
            rng=rng,
            max_depth=config.max_depth,
            min_depth=config.min_depth,
            max_size=config.max_size,
            libraries=[],
            allow_divergent=False
        )
        self.gold_reducer = GraphReducer(max_steps=config.max_steps)
        self.tree_reducer = TreeReducer(max_steps=config.max_steps)

        # Statistics
        self.comparisons: List[ComparisonMetrics] = []

    def span_to_path(self, render_result: RenderResult,
                     char_start: int, char_end: int) -> Optional[List[int]]:
        # Convert character span to structural path using render spans.
        # Find the node whose span best overlaps with the predicted span.

        best_node_id = None
        best_overlap = 0

        for node_id, (node_start, node_end) in render_result.spans.items():
            # Calculate overlap
            overlap_start = max(char_start, node_start)
            overlap_end = min(char_end, node_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_node_id = node_id

        if best_node_id is None:
            return None

        # Convert node_id to path
        # Root is 1, left child is 2*n+1, right child is 2*n+2
        # To get path, we work backwards from node_id to root
        path = []
        node = best_node_id

        while node > 1:
            if node % 2 == 1:  # Left child (odd)
                path.append(0)
                node = (node - 1) // 2
            else:  # Right child (even)
                path.append(1)
                node = (node - 2) // 2

        # Path was built backwards, reverse it
        path.reverse()
        return path

    @torch.no_grad()
    def predict_redex(self, term: Term) -> Tuple[Optional[List[int]], bool, RenderResult]:
        # Predict redex path using the model.
        # Returns (path, is_normal_form, render_result)

        # Render term with span tracking
        render_result = Renderer.to_debruijn_with_spans(term)
        term_str = render_result.string

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
        nf_logit = outputs['nf_logits'][0].item()

        # Check if model predicts normal form
        nf_prob = torch.sigmoid(torch.tensor(nf_logit)).item()
        if nf_prob > 0.5:
            return None, True, render_result

        # Get start and end positions
        start_probs = F.softmax(start_logits, dim=0)
        end_probs = F.softmax(end_logits, dim=0)

        start_idx = int(start_probs.argmax().item())
        end_idx = int(end_probs.argmax().item())

        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx

        # Convert token indices to character offsets
        # Skip BOS token (index 0)
        if start_idx == 0 or start_idx >= len(offsets):
            return None, False, render_result
        if end_idx == 0 or end_idx >= len(offsets):
            return None, False, render_result

        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]

        # Convert character span to structural path
        path = self.span_to_path(render_result, start_char, end_char)

        return path, False, render_result

    def reduce_with_model(self, term: Term) -> ReductionTrace:
        # Reduce a term using model predictions for redex selection.

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
            redex_path, is_nf, render_result = self.predict_redex(current_term)

            # Store step with character span (if available)
            char_span = None
            if redex_path and len(redex_path) > 0:
                # Find the span corresponding to this path
                node_id = self._path_to_node_id(redex_path)
                if node_id in render_result.spans:
                    char_span = render_result.spans[node_id]

            steps.append((term_str, char_span))

            if is_nf or redex_path is None:
                # Model predicts normal form
                return ReductionTrace(
                    strategy='model',
                    steps=steps,
                    converged=True,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    tokens_per_step=tokens_per_step
                )

            # Verify the path points to a valid redex
            if not self._is_valid_redex_path(current_term, redex_path):
                # Invalid prediction - treat as converged
                return ReductionTrace(
                    strategy='model',
                    steps=steps,
                    converged=True,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    tokens_per_step=tokens_per_step
                )

            # Reduce at predicted path
            try:
                current_term = self.tree_reducer._apply_reduction(current_term, redex_path)
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

    def _path_to_node_id(self, path: List[int]) -> int:
        # Convert path to node_id for span lookup
        node_id = 1
        for direction in path:
            if direction == 0:
                node_id = node_id * 2 + 1
            else:
                node_id = node_id * 2 + 2
        return node_id

    def _is_valid_redex_path(self, term: Term, path: List[int]) -> bool:
        # Check if path points to a valid redex (APP of ABS)
        try:
            current = term
            for direction in path:
                if direction == 0:
                    if current.type == TermType.ABS:
                        current = current.body
                    elif current.type == TermType.APP:
                        current = current.left
                    else:
                        return False
                else:  # direction == 1
                    if current.type == TermType.APP:
                        current = current.right
                    else:
                        return False

            # Check if we landed on a redex
            return (current.type == TermType.APP and
                    current.left and
                    current.left.type == TermType.ABS)
        except:
            return False

    def reduce_with_gold(self, term: Term) -> ReductionTrace:
        # Reduce using Levy graph reduction (gold standard).
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
                render_result = Renderer.to_debruijn_with_spans(term_obj)
                node_id = self._path_to_node_id(redex_path)
                if node_id in render_result.spans:
                    redex_span = render_result.spans[node_id]

            steps.append((term_str, redex_span))

        return ReductionTrace(
            strategy='gold',
            steps=steps,
            converged=not exceeded_max,
            total_steps=len(trace),
            total_tokens=total_tokens,
            tokens_per_step=tokens_per_step
        )

    def compare_strategies(self, term: Term) -> ComparisonMetrics:
        # Run both strategies and compare results.
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
        # Print detailed comparison.
        print(f"\n  Model: {metrics.model_trace.total_steps} steps, "
              f"{metrics.model_trace.total_tokens} tokens")
        print(f"  Gold:  {metrics.gold_trace.total_steps} steps, "
              f"{metrics.gold_trace.total_tokens} tokens")
        print(f"  Δ Steps: {metrics.step_difference:+d}, "
              f"Δ Tokens: {metrics.token_difference:+d}")
        print(f"  Diverged: {metrics.diverged} (step {metrics.divergence_step})")
        print(f"  Same NF: {metrics.same_normal_form}")

    def run_investigation(self):
        # Run full investigation on generated terms.
        print(f"\n{'='*70}")
        print(f"Lambda Calculus Reduction Strategy Investigation")
        print(f"{'='*70}\n")
        print(f"Generating {self.config.num_terms} terms...")
        print(f"Depth range: [{self.config.min_depth}, {self.config.max_depth}]")
        print(f"Max size: {self.config.max_size}")
        print(f"Max steps: {self.config.max_steps}\n")

        # Generate terms
        terms = []
        attempts = 0
        max_attempts = self.config.num_terms * 10

        while len(terms) < self.config.num_terms and attempts < max_attempts:
            term = self.term_gen.generate()
            if term:
                terms.append(term)
                if (len(terms)) % 20 == 0:
                    print(f"  Generated {len(terms)}/{self.config.num_terms} terms...")
            attempts += 1

        if len(terms) < self.config.num_terms:
            print(f"Warning: Only generated {len(terms)} terms after {attempts} attempts")

        print(f"\nRunning dual reduction (model vs gold)...\n")

        # Run comparisons
        for i, term in enumerate(terms):
            if not self.config.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(terms)} terms...")

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
        # Print summary statistics.
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
        # Save detailed results to JSON.
        if self.config.output_dir is None:
            return

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
    parser.add_argument('--min-depth', type=int, default=2,
                       help='Minimum term depth')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum term depth')
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
