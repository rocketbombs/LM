#!/usr/bin/env python3
#
# Lambda Calculus Reduction Inference Engine
#
# Loads a trained model checkpoint and compares its reduction strategy
# against classical normal-order reduction. Analyzes divergence points,
# efficiency metrics, and emergent optimization behavior.
#
# Usage:
#   # Basic comparison with 100 terms
#   python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt --num-terms 100
#
#   # Detailed output with verbose mode
#   python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt \
#       --num-terms 50 --verbose
#
#   # Save results to JSON for further analysis
#   python lambda_infer.py --checkpoint runs/levy700m/checkpoints/step_10000.pt \
#       --num-terms 200 --output-dir results/comparison_10k

import argparse
import json
import math
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
from lambda_gen import (TermGenerator, TreeReducer, Term, TermType,
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
    nf_confidence_threshold: float = 0.999


@dataclass
class ReductionTrace:
    # Trace of a reduction sequence.
    strategy: str  # 'neural' or 'classical'
    # steps removed to save memory - for 1000+ step reductions, storing all
    # intermediate term strings (300+ chars each) wastes 300KB+ per trace
    # We only need final_nf_str for comparison
    converged: bool
    total_steps: int
    total_tokens: int
    final_nf_str: str = ""  # Final normal form as string for comparison
    final_term: Optional[Term] = None  # The final term object (set to None after comparison)
    divergence_step: Optional[int] = None  # When it diverged from classical


@dataclass
class ComparisonMetrics:
    # Metrics comparing neural vs classical reduction.
    term: str
    neural_trace: ReductionTrace
    classical_trace: ReductionTrace

    # Efficiency metrics
    neural_faster: bool
    step_difference: int  # Negative = neural is faster
    token_difference: int  # Negative = neural uses fewer tokens

    # Divergence analysis
    diverged: bool
    divergence_step: Optional[int]

    # Correctness
    same_normal_form: bool

    # Optional fields with defaults
    divergence_context: Optional[str] = None
    neural_nf: str = ""
    classical_nf: str = ""


class InferenceEngine:
    # Engine for running inference and comparing reduction strategies.

    def __init__(self, config: InferenceConfig):
        self.config = config
        torch.manual_seed(config.seed)

        if not (0.5 < config.nf_confidence_threshold < 1.0):
            raise ValueError("nf_confidence_threshold must be between 0.5 and 1.0 (exclusive).")
        self._nf_logit_threshold = math.log(
            config.nf_confidence_threshold / (1.0 - config.nf_confidence_threshold)
        )

        # Setup device
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")

        # Load checkpoint
        print(f"\nLoading checkpoint from: {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint, map_location=self.device)

        # Extract training config
        train_config_raw = checkpoint.get('config', None)
        if train_config_raw is None:
            # Try to load from config.json in same directory
            config_path = Path(config.checkpoint).parent / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                    self.train_config = TrainingConfig(**config_dict)
            else:
                raise ValueError("Could not find training config in checkpoint or config.json")
        elif isinstance(train_config_raw, dict):
            # Config is stored as dict, convert to TrainingConfig
            self.train_config = TrainingConfig(**train_config_raw)
        else:
            # Already a TrainingConfig object
            self.train_config = train_config_raw

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
        # Use tree reducer for both classical and neural (normal-order strategy)
        self.classical_reducer = TreeReducer(max_steps=config.max_steps)
        self.neural_reducer = TreeReducer(max_steps=config.max_steps)

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
        if nf_logit >= 0:
            exp_term = math.exp(-nf_logit)
            nf_prob = 1.0 / (1.0 + exp_term)
        else:
            exp_term = math.exp(nf_logit)
            nf_prob = exp_term / (1.0 + exp_term)

        has_redex = self._has_redex(term)
        is_confident_nf = nf_logit >= self._nf_logit_threshold

        if self.config.verbose:
            print(
                f"    Model NF prediction: {nf_prob:.6f} "
                f"(threshold {self.config.nf_confidence_threshold:.6f}) | "
                f"Actual has redex: {has_redex}"
            )
            if is_confident_nf and has_redex:
                print("    Warning: NF head is overconfident; falling back to reduction.")

        if not has_redex:
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

    def reduce_with_neural(self, term: Term) -> ReductionTrace:
        # Reduce a term using neural model predictions for redex selection.

        current_term = term
        total_tokens = 0

        for step_num in range(self.config.max_steps):
            term_str = Renderer.to_debruijn_with_spans(current_term).string
            term_tokens = len(term_str)
            total_tokens += term_tokens

            # Get model prediction
            redex_path, is_nf, render_result = self.predict_redex(current_term)

            # Accept neural model's NF prediction as-is
            if is_nf:
                final_nf_str = render_result.string
                return ReductionTrace(
                    strategy='neural',
                    converged=True,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    final_nf_str=final_nf_str,
                    final_term=current_term
                )

            chosen_path = redex_path
            fallback_reason = None

            if chosen_path is None:
                fallback_reason = "no_span"
            elif not self._is_valid_redex_path(current_term, chosen_path):
                fallback_reason = "invalid_span"

            if fallback_reason is not None:
                chosen_path = self.neural_reducer._find_leftmost_outermost(current_term)
                if self.config.verbose:
                    print(f"    Falling back to classical span ({fallback_reason}).")

            if chosen_path is None:
                has_redexes = self._has_redex(current_term)
                final_nf_str = render_result.string
                return ReductionTrace(
                    strategy='neural',
                    converged=not has_redexes,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    final_nf_str=final_nf_str,
                    final_term=current_term
                )

            # Reduce at predicted path
            try:
                current_term = self.neural_reducer._apply_reduction(current_term, chosen_path)
            except Exception as e:
                if self.config.verbose:
                    print(f"Reduction failed at step {step_num}: {e}")
                final_nf_str = Renderer.to_debruijn_with_spans(current_term).string
                return ReductionTrace(
                    strategy='neural',
                    converged=False,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    final_nf_str=final_nf_str,
                    final_term=current_term
                )
            finally:
                # Drop reference eagerly to let GC reclaim span maps
                render_result = None

        # Exceeded max steps
        final_nf_str = Renderer.to_debruijn_with_spans(current_term).string
        return ReductionTrace(
            strategy='neural',
            converged=False,
            total_steps=self.config.max_steps,
            total_tokens=total_tokens,
            final_nf_str=final_nf_str,
            final_term=current_term
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
                        if current.body is None:
                            return False
                        current = current.body
                    elif current.type == TermType.APP:
                        if current.left is None:
                            return False
                        current = current.left
                    else:
                        return False
                else:  # direction == 1
                    if current.type == TermType.APP:
                        if current.right is None:
                            return False
                        current = current.right
                    else:
                        return False

            # Check if we landed on a redex
            if current.type != TermType.APP:
                return False
            if current.left is None:
                return False
            if current.left.type != TermType.ABS:
                return False
            return True
        except:
            return False

    def reduce_with_classical(self, term: Term) -> ReductionTrace:
        # Reduce using classical normal-order tree reduction (baseline).
        trace, exceeded_max = self.classical_reducer.reduce(term)

        total_tokens = 0
        final_term_obj = None
        final_nf_str = ""

        # Process trace immediately, don't store intermediate strings
        trace_len = len(trace)
        for i, (term_obj, redex_path) in enumerate(trace):
            # Validate that term_obj is actually a Term
            if not isinstance(term_obj, Term):
                raise TypeError(f"Trace step {i}: expected Term, got {type(term_obj).__name__}. "
                              f"This is a bug in TreeReducer.")

            term_str = Renderer.to_debruijn_with_spans(term_obj).string
            term_tokens = len(term_str)
            total_tokens += term_tokens

            # Keep only the final term and its string
            if i == trace_len - 1:
                final_term_obj = term_obj
                final_nf_str = term_str

        # Explicitly delete trace to free memory immediately
        del trace

        return ReductionTrace(
            strategy='classical',
            converged=not exceeded_max,
            total_steps=trace_len,
            total_tokens=total_tokens,
            final_nf_str=final_nf_str,
            final_term=final_term_obj
        )

    def _has_redex(self, term: Term) -> bool:
        # Check if term contains any redex (is not in normal form)
        if not isinstance(term, Term):
            raise TypeError(f"_has_redex expected Term, got {type(term).__name__}")
        if term.type == TermType.APP and term.left and term.left.type == TermType.ABS:
            return True

        if term.type == TermType.ABS and term.body:
            return self._has_redex(term.body)
        elif term.type == TermType.APP:
            if term.left and self._has_redex(term.left):
                return True
            if term.right and self._has_redex(term.right):
                return True

        return False

    def _terms_equal(self, t1: Term, t2: Term) -> bool:
        # Structural equality check for terms
        if t1.type != t2.type:
            return False

        if t1.type == TermType.VAR:
            return t1.var == t2.var
        elif t1.type == TermType.ABS:
            if t1.body is None or t2.body is None:
                return t1.body == t2.body
            return self._terms_equal(t1.body, t2.body)
        else:  # APP
            if t1.left is None or t2.left is None or t1.right is None or t2.right is None:
                return (t1.left == t2.left and t1.right == t2.right)
            return (self._terms_equal(t1.left, t2.left) and
                   self._terms_equal(t1.right, t2.right))

    def compare_strategies(self, term: Term) -> ComparisonMetrics:
        # Run both strategies and compare results.
        if not isinstance(term, Term):
            raise TypeError(f"compare_strategies requires Term, got {type(term).__name__}")

        term_str = Renderer.to_debruijn_with_spans(term).string

        if self.config.verbose:
            print(f"\nTerm: {term_str}")

        # Run both reductions
        neural_trace = self.reduce_with_neural(term)
        classical_trace = self.reduce_with_classical(term)

        # Extract normal forms (strings and terms)
        neural_nf = neural_trace.final_nf_str
        classical_nf = classical_trace.final_nf_str

        # Check if both are actually in normal form
        neural_is_nf = (neural_trace.final_term is None or
                      not self._has_redex(neural_trace.final_term))
        classical_is_nf = (classical_trace.final_term is None or
                     not self._has_redex(classical_trace.final_term))

        # Check structural equality if both reached NF
        structurally_equal = False
        if (neural_trace.final_term is not None and
            classical_trace.final_term is not None and
            neural_is_nf and classical_is_nf):
            structurally_equal = self._terms_equal(neural_trace.final_term,
                                                   classical_trace.final_term)

        # Check divergence - we can't determine exact step anymore since we don't store
        # intermediate terms, but we can infer divergence from different step counts or NFs
        diverged = (neural_trace.total_steps != classical_trace.total_steps) or (neural_nf != classical_nf)
        divergence_step = None  # Can't determine exact step without storing intermediate terms

        # Compute metrics
        step_diff = neural_trace.total_steps - classical_trace.total_steps
        token_diff = neural_trace.total_tokens - classical_trace.total_tokens

        # Consider "same" if string match OR structural equality
        same_nf = (neural_nf == classical_nf) or structurally_equal

        context = None
        if not neural_is_nf:
            context = "neural_not_nf"
        elif not classical_is_nf:
            context = "classical_not_nf"
        elif not same_nf:
            context = "different_nf"

        metrics = ComparisonMetrics(
            term=term_str,
            neural_trace=neural_trace,
            classical_trace=classical_trace,
            neural_faster=step_diff < 0,
            step_difference=step_diff,
            token_difference=token_diff,
            diverged=diverged,
            divergence_step=divergence_step,
            same_normal_form=same_nf,
            divergence_context=context,
            neural_nf=neural_nf,
            classical_nf=classical_nf
        )

        if self.config.verbose:
            self._print_comparison(metrics)

        # Free memory: We no longer need the Term objects after comparison
        # They've served their purpose for structural equality checking
        neural_trace.final_term = None
        classical_trace.final_term = None

        return metrics

    def _print_comparison(self, metrics: ComparisonMetrics):
        # Print detailed comparison.
        print(f"\n  Neural: {metrics.neural_trace.total_steps} steps, "
              f"{metrics.neural_trace.total_tokens} tokens")
        print(f"  Classical:  {metrics.classical_trace.total_steps} steps, "
              f"{metrics.classical_trace.total_tokens} tokens")
        print(f"  Δ Steps: {metrics.step_difference:+d}, "
              f"Δ Tokens: {metrics.token_difference:+d}")
        print(f"  Diverged: {metrics.diverged} (step {metrics.divergence_step})")
        print(f"  Same NF: {metrics.same_normal_form}")

    def run_comparison(self):
        # Run full comparison on generated terms.
        print(f"\n{'='*70}")
        print(f"Lambda Calculus Neural vs Classical Comparison")
        print(f"{'='*70}\n")
        print(f"Generating {self.config.num_terms} terms...")
        print(f"Depth range: [{self.config.min_depth}, {self.config.max_depth}]")
        print(f"Max size: {self.config.max_size}")
        print(f"Max steps: {self.config.max_steps}\n")

        max_attempts = self.config.num_terms * 10
        attempts = 0
        generated = 0
        processed = 0

        print(f"\nRunning dual reduction (neural vs classical)...\n")

        while processed < self.config.num_terms and attempts < max_attempts:
            term = self.term_gen.generate()
            attempts += 1

            if term is None:
                continue

            generated += 1
            if generated % 20 == 0:
                print(f"  Generated {generated}/{self.config.num_terms} candidate terms...")

            try:
                metrics = self.compare_strategies(term)
            except Exception as e:
                if self.config.verbose:
                    print(f"Error processing generated term {generated}: {e}")
            else:
                self.comparisons.append(metrics)
                processed += 1

                if not self.config.verbose and processed % 10 == 0:
                    print(f"  Processed {processed}/{self.config.num_terms} terms...")
            finally:
                # Explicitly drop reference to the last term to help GC in long runs
                del term

        if processed < self.config.num_terms:
            print(
                f"Warning: Completed {processed}/{self.config.num_terms} terms "
                f"after {attempts} attempts"
            )

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
        print(f"COMPARISON SUMMARY")
        print(f"{'='*70}\n")

        total = len(self.comparisons)

        # Convergence
        neural_converged = sum(1 for c in self.comparisons if c.neural_trace.converged)
        classical_converged = sum(1 for c in self.comparisons if c.classical_trace.converged)

        print(f"Convergence:")
        print(f"  Neural: {neural_converged}/{total} ({100*neural_converged/total:.1f}%)")
        print(f"  Classical: {classical_converged}/{total} ({100*classical_converged/total:.1f}%)")

        # Correctness
        same_nf = sum(1 for c in self.comparisons if c.same_normal_form)
        print(f"\nAccuracy (Correct NF):")
        print(f"  Same normal form: {same_nf}/{total} ({100*same_nf/total:.1f}%)")

        # Breakdown of mismatches
        neural_not_nf = sum(1 for c in self.comparisons
                          if c.divergence_context == "neural_not_nf")
        classical_not_nf = sum(1 for c in self.comparisons
                         if c.divergence_context == "classical_not_nf")
        different_nf = sum(1 for c in self.comparisons
                          if c.divergence_context == "different_nf")

        if neural_not_nf + classical_not_nf + different_nf > 0:
            print(f"\nNormal Form Analysis:")
            print(f"  Neural stopped early (still has redex): {neural_not_nf}/{total} "
                  f"({100*neural_not_nf/total:.1f}%)")
            print(f"  Classical not in NF (unexpected): {classical_not_nf}/{total} "
                  f"({100*classical_not_nf/total:.1f}%)")
            print(f"  Both in NF but different: {different_nf}/{total} "
                  f"({100*different_nf/total:.1f}%)")

        # Divergence
        diverged = sum(1 for c in self.comparisons if c.diverged)
        print(f"\nPath Divergence:")
        print(f"  Different paths: {diverged}/{total} ({100*diverged/total:.1f}%)")

        if diverged > 0:
            div_steps = [c.divergence_step for c in self.comparisons
                        if c.divergence_step is not None]
            if div_steps:
                avg_div_step = sum(div_steps) / len(div_steps)
                print(f"  Average divergence step: {avg_div_step:.1f}")

        # Efficiency
        neural_faster_count = sum(1 for c in self.comparisons if c.neural_faster)
        print(f"\nEfficiency (Steps):")
        print(f"  Neural faster: {neural_faster_count}/{total} "
              f"({100*neural_faster_count/total:.1f}%)")

        step_diffs = [c.step_difference for c in self.comparisons]
        token_diffs = [c.token_difference for c in self.comparisons]

        avg_step_diff = sum(step_diffs) / len(step_diffs)
        avg_token_diff = sum(token_diffs) / len(token_diffs)

        print(f"  Average step difference: {avg_step_diff:+.2f} (negative = neural faster)")
        print(f"  Average token difference: {avg_token_diff:+.2f}")

        # Step efficiency
        neural_avg_steps = sum(c.neural_trace.total_steps for c in self.comparisons) / total
        classical_avg_steps = sum(c.classical_trace.total_steps for c in self.comparisons) / total
        efficiency = (neural_avg_steps / classical_avg_steps) * 100 if classical_avg_steps > 0 else 100

        print(f"\n  Neural avg steps: {neural_avg_steps:.2f}")
        print(f"  Classical avg steps: {classical_avg_steps:.2f}")
        print(f"  Step efficiency: {efficiency:.1f}% (neural uses {efficiency:.1f}% of classical steps)")

        # Token throughput
        neural_avg_tokens = sum(c.neural_trace.total_tokens for c in self.comparisons) / total
        classical_avg_tokens = sum(c.classical_trace.total_tokens for c in self.comparisons) / total

        print(f"\nToken Throughput:")
        print(f"  Neural avg total tokens: {neural_avg_tokens:.1f}")
        print(f"  Classical avg total tokens: {classical_avg_tokens:.1f}")

        # Most interesting cases
        print(f"\nMost Interesting Cases:")

        # Cases where neural is significantly faster
        faster_cases = sorted([c for c in self.comparisons if c.step_difference < -2],
                            key=lambda c: c.step_difference)[:3]

        if faster_cases:
            print(f"\n  Neural significantly faster:")
            for c in faster_cases:
                print(f"    {c.step_difference:+d} steps | Term: {c.term[:60]}...")
                if self.config.verbose or len(faster_cases) <= 1:
                    print(f"      Neural NF: {c.neural_nf[:50]}...")
                    print(f"      Classical NF:  {c.classical_nf[:50]}...")

        # Cases where neural is significantly slower
        slower_cases = sorted([c for c in self.comparisons if c.step_difference > 2],
                            key=lambda c: -c.step_difference)[:3]

        if slower_cases:
            print(f"\n  Neural significantly slower:")
            for c in slower_cases:
                print(f"    {c.step_difference:+d} steps | Term: {c.term[:60]}...")

        # Cases with different normal forms (potential neural errors)
        different_nf = [c for c in self.comparisons if not c.same_normal_form]
        if different_nf:
            print(f"\n  Different normal forms (errors): {len(different_nf)}")
            for c in different_nf[:2]:
                print(f"    Term: {c.term[:50]}...")
                print(f"      Neural NF: {c.neural_nf[:50]}...")
                print(f"      Classical NF:  {c.classical_nf[:50]}...")

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
                    'neural_steps': c.neural_trace.total_steps,
                    'classical_steps': c.classical_trace.total_steps,
                    'neural_tokens': c.neural_trace.total_tokens,
                    'classical_tokens': c.classical_trace.total_tokens,
                    'step_difference': c.step_difference,
                    'token_difference': c.token_difference,
                    'diverged': c.diverged,
                    'divergence_step': c.divergence_step,
                    'same_nf': c.same_normal_form,
                    'neural_nf': c.neural_nf,
                    'classical_nf': c.classical_nf,
                }
                for c in self.comparisons
            ]
        }

        output_file = output_dir / 'comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare neural vs classical lambda calculus reduction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', required=True,
                       help='Path to neural model checkpoint')
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
    parser.add_argument('--nf-confidence-threshold', type=float, default=0.999,
                        help='Minimum probability required to declare normal form')

    args = parser.parse_args()
    config = InferenceConfig(**vars(args))

    # Run comparison
    engine = InferenceEngine(config)
    engine.run_comparison()


if __name__ == '__main__':
    main()
