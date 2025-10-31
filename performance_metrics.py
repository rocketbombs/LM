#!/usr/bin/env python3
#
# Comprehensive Performance Metrics for Lambda Calculus Model
#
# This script performs a detailed investigation of model performance with extensive
# metrics tracking. It analyzes term types, reduction strategies, speed, and emergent
# behaviors to provide a complete picture of model capabilities.
#
# Usage:
#   python performance_metrics.py --checkpoint runs/levy700m/checkpoints/step_10000.pt \
#       --num-terms 200 --output-dir results/performance_analysis

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import statistics

import torch
import torch.nn.functional as F

from lambda_train import LambdaSpanPredictor, LambdaTokenizer, TrainingConfig
from lambda_gen import (TermGenerator, GraphReducer, TreeReducer, Term, TermType,
                         reference_substitute, Renderer, RenderResult)


@dataclass
class TermCharacteristics:
    """Detailed characteristics of a lambda term."""
    size: int  # Total number of nodes
    depth: int  # Maximum nesting depth
    num_vars: int
    num_abs: int
    num_apps: int
    num_redexes: int  # Total redexes in term
    initial_redex_depth: Optional[int]  # Depth of first redex
    initial_redex_path: Optional[List[int]]
    complexity_score: float  # size * depth


@dataclass
class ReductionStepMetrics:
    """Metrics for a single reduction step."""
    step_num: int
    term_size: int
    term_depth: int
    redex_depth: int  # How deep in tree is the redex
    redex_path: List[int]
    inference_time_ms: float  # Time to predict this step
    nf_confidence: float  # Model's confidence in normal form
    span_start: Optional[int]
    span_end: Optional[int]
    term_string: str


@dataclass
class DetailedReductionTrace:
    """Enhanced reduction trace with step-by-step metrics."""
    strategy: str  # 'model' or 'gold'
    initial_characteristics: TermCharacteristics
    step_metrics: List[ReductionStepMetrics]
    converged: bool
    total_steps: int
    total_tokens: int
    total_inference_time_ms: float
    avg_inference_time_ms: float
    final_term: Optional[Term]
    size_evolution: List[int]  # Term size at each step
    depth_evolution: List[int]  # Term depth at each step


@dataclass
class PathAnalysis:
    """Analysis of reduction path patterns."""
    path_depth_distribution: Dict[int, int]  # depth -> count
    path_patterns: Counter  # Common path prefixes
    left_vs_right_bias: Dict[str, int]  # Count of left vs right choices
    root_reductions: int  # Reductions at root
    deep_reductions: int  # Reductions deeper than depth 3


@dataclass
class PerformanceComparison:
    """Comprehensive comparison of model vs gold performance."""
    term_id: int
    initial_term: str
    initial_characteristics: TermCharacteristics

    # Traces
    model_trace: DetailedReductionTrace
    gold_trace: Optional[DetailedReductionTrace]

    # Correctness
    same_normal_form: bool
    model_correct: bool  # Did model reach correct NF

    # Efficiency
    model_faster_steps: bool
    step_difference: int  # Negative = model faster
    token_difference: int
    time_difference_ms: float  # Negative = model faster

    # Divergence
    diverged: bool
    divergence_step: Optional[int]
    divergence_type: Optional[str]  # 'path_choice', 'early_stop', 'invalid_pred'

    # Strategy comparison
    model_path_analysis: Optional[PathAnalysis] = None
    gold_path_analysis: Optional[PathAnalysis] = None


@dataclass
class AggregatedMetrics:
    """Aggregated statistics across all test terms."""
    # Overall statistics
    total_terms: int
    successful_terms: int

    # Convergence
    model_convergence_rate: float
    gold_convergence_rate: float

    # Correctness
    accuracy: float  # % terms reaching correct NF
    same_path_rate: float  # % terms following exact same path as gold

    # Efficiency
    model_faster_rate: float  # % terms where model is faster
    avg_step_difference: float
    avg_token_difference: float
    avg_time_per_inference_ms: float

    # Divergence
    divergence_rate: float
    avg_divergence_step: Optional[float]
    divergence_type_distribution: Counter

    # Term characteristics vs performance
    performance_by_size: Dict[str, Dict[str, float]]  # size_bucket -> metrics
    performance_by_depth: Dict[str, Dict[str, float]]  # depth_bucket -> metrics
    performance_by_complexity: Dict[str, Dict[str, float]]

    # Path pattern analysis
    model_path_patterns: Counter  # Most common path patterns
    gold_path_patterns: Counter
    unique_model_patterns: List[str]  # Patterns only model uses

    # Speed analysis
    terms_by_speedup: List[Tuple[int, float]]  # (term_id, speedup_ratio)
    fastest_term_types: List[Dict[str, Any]]  # Characteristics of fastest reductions
    slowest_term_types: List[Dict[str, Any]]

    # Error analysis
    error_cases: List[Dict[str, Any]]
    common_failure_patterns: Counter


class PerformanceAnalyzer:
    """Comprehensive performance analysis engine."""

    def __init__(self, checkpoint: str, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_path = checkpoint

        # Setup device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        # Load model
        print(f"\nLoading checkpoint: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=self.device)

        # Extract training config
        train_config_raw = checkpoint_data.get('config')
        if train_config_raw is None:
            config_path = Path(checkpoint).parent / 'config.json'
            with open(config_path) as f:
                self.train_config = TrainingConfig(**json.load(f))
        elif isinstance(train_config_raw, dict):
            self.train_config = TrainingConfig(**train_config_raw)
        else:
            self.train_config = train_config_raw

        print(f"Model: {self.train_config.d_model}d × {self.train_config.n_layers}L")

        # Initialize tokenizer and model
        self.tokenizer = LambdaTokenizer()
        self.model = LambdaSpanPredictor(self.train_config, self.tokenizer.vocab_size)
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Checkpoint step: {checkpoint_data.get('step', 'unknown')}")

        # Initialize generators and reducers
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        rng = random.Random(seed)

        self.term_gen = TermGenerator(
            rng=rng,
            max_depth=config.get('max_depth', 10),
            min_depth=config.get('min_depth', 2),
            max_size=config.get('max_size', 100),
            libraries=[],
            allow_divergent=False
        )

        max_steps = config.get('max_steps', 1000)
        wall_clock_limit = config.get('wall_clock_limit_ms', 100.0)
        self.gold_reducer = GraphReducer(
            max_steps=max_steps,
            wall_clock_limit_ms=wall_clock_limit
        )
        self.tree_reducer = TreeReducer(max_steps=max_steps)

        # Storage for results
        self.comparisons: List[PerformanceComparison] = []

    def analyze_term_characteristics(self, term: Term) -> TermCharacteristics:
        """Extract detailed characteristics from a term."""

        def count_nodes(t: Term) -> Tuple[int, int, int, int]:
            """Returns (vars, abs, apps, total)"""
            if t.type == TermType.VAR:
                return (1, 0, 0, 1)
            elif t.type == TermType.ABS:
                if t.body:
                    v, a, ap, tot = count_nodes(t.body)
                    return (v, a + 1, ap, tot + 1)
                return (0, 1, 0, 1)
            else:  # APP
                left_v, left_a, left_ap, left_tot = (0, 0, 0, 0) if not t.left else count_nodes(t.left)
                right_v, right_a, right_ap, right_tot = (0, 0, 0, 0) if not t.right else count_nodes(t.right)
                return (left_v + right_v, left_a + right_a, left_ap + right_ap + 1,
                       left_tot + right_tot + 1)

        def compute_depth(t: Term) -> int:
            """Compute maximum nesting depth."""
            if t.type == TermType.VAR:
                return 1
            elif t.type == TermType.ABS:
                return 1 + (compute_depth(t.body) if t.body else 0)
            else:  # APP
                left_d = compute_depth(t.left) if t.left else 0
                right_d = compute_depth(t.right) if t.right else 0
                return 1 + max(left_d, right_d)

        def find_redexes(t: Term, path: List[int] = []) -> List[Tuple[List[int], int]]:
            """Find all redexes and their depths. Returns [(path, depth)]"""
            redexes = []
            depth = len(path)

            # Check if current position is a redex
            if t.type == TermType.APP and t.left and t.left.type == TermType.ABS:
                redexes.append((path.copy(), depth))

            # Recurse
            if t.type == TermType.ABS and t.body:
                redexes.extend(find_redexes(t.body, path + [0]))
            elif t.type == TermType.APP:
                if t.left:
                    redexes.extend(find_redexes(t.left, path + [0]))
                if t.right:
                    redexes.extend(find_redexes(t.right, path + [1]))

            return redexes

        num_vars, num_abs, num_apps, size = count_nodes(term)
        depth = compute_depth(term)
        redexes = find_redexes(term)

        initial_redex_path = None
        initial_redex_depth = None
        if redexes:
            # Normal order: leftmost-outermost (minimum depth, then leftmost)
            redexes.sort(key=lambda x: (x[1], x[0]))
            initial_redex_path = redexes[0][0]
            initial_redex_depth = redexes[0][1]

        return TermCharacteristics(
            size=size,
            depth=depth,
            num_vars=num_vars,
            num_abs=num_abs,
            num_apps=num_apps,
            num_redexes=len(redexes),
            initial_redex_depth=initial_redex_depth,
            initial_redex_path=initial_redex_path,
            complexity_score=size * depth
        )

    def _path_to_node_id(self, path: List[int]) -> int:
        """Convert path to node_id for span lookup."""
        node_id = 1
        for direction in path:
            node_id = node_id * 2 + (1 if direction == 0 else 2)
        return node_id

    def _is_valid_redex_path(self, term: Term, path: List[int]) -> bool:
        """Check if path points to a valid redex."""
        try:
            current = term
            for direction in path:
                if direction == 0:
                    if current.type == TermType.ABS:
                        if not current.body:
                            return False
                        current = current.body
                    elif current.type == TermType.APP:
                        if not current.left:
                            return False
                        current = current.left
                    else:
                        return False
                else:  # direction == 1
                    if current.type == TermType.APP:
                        if not current.right:
                            return False
                        current = current.right
                    else:
                        return False

            # Check if we're at a redex
            return (current.type == TermType.APP and
                   current.left is not None and
                   current.left.type == TermType.ABS)
        except:
            return False

    @torch.no_grad()
    def predict_redex_with_metrics(self, term: Term, step_num: int) -> Tuple[
        Optional[List[int]], bool, RenderResult, float, float]:
        """Predict redex and collect timing + confidence metrics.

        Returns: (path, is_nf, render_result, inference_time_ms, nf_confidence)
        """
        start_time = time.perf_counter()

        # Render term
        render_result = Renderer.to_debruijn_with_spans(term)
        term_str = render_result.string

        # Tokenize
        token_ids, offsets = self.tokenizer.encode(term_str, add_special=True)

        # Prepare batch
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Get predictions
        start_logits = outputs['start_logits'][0]
        end_logits = outputs['end_logits'][0]
        nf_logit = outputs['nf_logits'][0].item()

        # NF prediction
        nf_prob = torch.sigmoid(torch.tensor(nf_logit)).item()
        if nf_prob > 0.5:
            return None, True, render_result, inference_time_ms, nf_prob

        # Span prediction
        start_probs = F.softmax(start_logits, dim=0)
        end_probs = F.softmax(end_logits, dim=0)

        start_idx = int(start_probs.argmax().item())
        end_idx = int(end_probs.argmax().item())

        if end_idx < start_idx:
            end_idx = start_idx

        # Convert to character offsets
        if start_idx == 0 or start_idx >= len(offsets):
            return None, False, render_result, inference_time_ms, nf_prob
        if end_idx == 0 or end_idx >= len(offsets):
            return None, False, render_result, inference_time_ms, nf_prob

        start_char = offsets[start_idx][0]
        end_char = offsets[end_idx][1]

        # Convert to path
        path = self._span_to_path(render_result, start_char, end_char)

        return path, False, render_result, inference_time_ms, nf_prob

    def _span_to_path(self, render_result: RenderResult,
                     char_start: int, char_end: int) -> Optional[List[int]]:
        """Convert character span to structural path."""
        best_node_id = None
        best_overlap = 0

        for node_id, (node_start, node_end) in render_result.spans.items():
            overlap_start = max(char_start, node_start)
            overlap_end = min(char_end, node_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_node_id = node_id

        if best_node_id is None:
            return None

        # Convert node_id to path
        path = []
        node = best_node_id
        while node > 1:
            if node % 2 == 1:
                path.append(0)
                node = (node - 1) // 2
            else:
                path.append(1)
                node = (node - 2) // 2

        path.reverse()
        return path

    def reduce_with_model_detailed(self, term: Term,
                                   initial_chars: TermCharacteristics) -> DetailedReductionTrace:
        """Reduce term with model, collecting detailed step metrics."""
        step_metrics: List[ReductionStepMetrics] = []
        current_term = term
        total_tokens = 0
        total_inference_time = 0.0
        size_evolution = []
        depth_evolution = []

        max_steps = self.config.get('max_steps', 1000)

        for step_num in range(max_steps):
            # Analyze current term
            chars = self.analyze_term_characteristics(current_term)
            size_evolution.append(chars.size)
            depth_evolution.append(chars.depth)
            total_tokens += chars.size

            # Predict next redex
            path, is_nf, render_result, inf_time, nf_conf = \
                self.predict_redex_with_metrics(current_term, step_num)

            total_inference_time += inf_time

            # Get span for this prediction
            span_start = span_end = None
            if path and len(path) >= 0:
                node_id = self._path_to_node_id(path)
                if node_id in render_result.spans:
                    span_start, span_end = render_result.spans[node_id]

            redex_depth = len(path) if path else 0

            # Record step metrics
            step_metrics.append(ReductionStepMetrics(
                step_num=step_num,
                term_size=chars.size,
                term_depth=chars.depth,
                redex_depth=redex_depth,
                redex_path=path if path else [],
                inference_time_ms=inf_time,
                nf_confidence=nf_conf,
                span_start=span_start,
                span_end=span_end,
                term_string=render_result.string
            ))

            # Check if done
            if is_nf or path is None:
                avg_inf_time = total_inference_time / (step_num + 1) if step_num >= 0 else 0
                return DetailedReductionTrace(
                    strategy='model',
                    initial_characteristics=initial_chars,
                    step_metrics=step_metrics,
                    converged=True,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    total_inference_time_ms=total_inference_time,
                    avg_inference_time_ms=avg_inf_time,
                    final_term=current_term,
                    size_evolution=size_evolution,
                    depth_evolution=depth_evolution
                )

            # Validate path
            if not self._is_valid_redex_path(current_term, path):
                avg_inf_time = total_inference_time / (step_num + 1)
                return DetailedReductionTrace(
                    strategy='model',
                    initial_characteristics=initial_chars,
                    step_metrics=step_metrics,
                    converged=True,  # Model stopped, accept as converged
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    total_inference_time_ms=total_inference_time,
                    avg_inference_time_ms=avg_inf_time,
                    final_term=current_term,
                    size_evolution=size_evolution,
                    depth_evolution=depth_evolution
                )

            # Apply reduction
            try:
                current_term = self.tree_reducer._apply_reduction(current_term, path)
            except Exception as e:
                avg_inf_time = total_inference_time / (step_num + 1)
                return DetailedReductionTrace(
                    strategy='model',
                    initial_characteristics=initial_chars,
                    step_metrics=step_metrics,
                    converged=False,
                    total_steps=step_num + 1,
                    total_tokens=total_tokens,
                    total_inference_time_ms=total_inference_time,
                    avg_inference_time_ms=avg_inf_time,
                    final_term=current_term,
                    size_evolution=size_evolution,
                    depth_evolution=depth_evolution
                )

        # Max steps exceeded
        avg_inf_time = total_inference_time / max_steps
        return DetailedReductionTrace(
            strategy='model',
            initial_characteristics=initial_chars,
            step_metrics=step_metrics,
            converged=False,
            total_steps=max_steps,
            total_tokens=total_tokens,
            total_inference_time_ms=total_inference_time,
            avg_inference_time_ms=avg_inf_time,
            final_term=current_term,
            size_evolution=size_evolution,
            depth_evolution=depth_evolution
        )

    def reduce_with_gold_detailed(self, term: Term,
                                  initial_chars: TermCharacteristics) -> DetailedReductionTrace:
        """Reduce with gold standard, collecting detailed metrics."""
        trace, exceeded_max, thunk_evals, thunk_hits, total_time_ms = self.gold_reducer.reduce(term)

        step_metrics: List[ReductionStepMetrics] = []
        total_tokens = 0
        size_evolution = []
        depth_evolution = []

        for step_num, (term_obj, redex_path, step_time_ms) in enumerate(trace):
            chars = self.analyze_term_characteristics(term_obj)
            size_evolution.append(chars.size)
            depth_evolution.append(chars.depth)
            total_tokens += chars.size

            render_result = Renderer.to_debruijn_with_spans(term_obj)

            span_start = span_end = None
            redex_depth = 0
            if redex_path:
                redex_depth = len(redex_path)
                node_id = self._path_to_node_id(redex_path)
                if node_id in render_result.spans:
                    span_start, span_end = render_result.spans[node_id]

            step_metrics.append(ReductionStepMetrics(
                step_num=step_num,
                term_size=chars.size,
                term_depth=chars.depth,
                redex_depth=redex_depth,
                redex_path=redex_path if redex_path else [],
                inference_time_ms=0.0,  # No inference for gold
                nf_confidence=1.0 if redex_path is None else 0.0,
                span_start=span_start,
                span_end=span_end,
                term_string=render_result.string
            ))

        final_term = trace[-1][0] if trace else term

        return DetailedReductionTrace(
            strategy='gold',
            initial_characteristics=initial_chars,
            step_metrics=step_metrics,
            converged=not exceeded_max,
            total_steps=len(trace),
            total_tokens=total_tokens,
            total_inference_time_ms=0.0,
            avg_inference_time_ms=0.0,
            final_term=final_term,
            size_evolution=size_evolution,
            depth_evolution=depth_evolution
        )

    def analyze_path_patterns(self, trace: DetailedReductionTrace) -> PathAnalysis:
        """Analyze path patterns in a reduction trace."""
        depth_dist: Dict[int, int] = defaultdict(int)
        patterns = Counter()
        left_right = Counter()
        root_reductions = 0
        deep_reductions = 0

        for step in trace.step_metrics:
            if not step.redex_path:
                continue

            depth = len(step.redex_path)
            depth_dist[depth] += 1

            if depth == 0:
                root_reductions += 1
                patterns['root'] += 1
            else:
                if depth > 3:
                    deep_reductions += 1

                # Track path patterns
                path_str = ''.join('L' if d == 0 else 'R' for d in step.redex_path[:3])
                patterns[path_str] += 1

                # Left vs right bias
                for d in step.redex_path:
                    left_right['left' if d == 0 else 'right'] += 1

        return PathAnalysis(
            path_depth_distribution=dict(depth_dist),
            path_patterns=patterns,
            left_vs_right_bias=dict(left_right),
            root_reductions=root_reductions,
            deep_reductions=deep_reductions
        )

    def _terms_equal(self, t1: Term, t2: Term) -> bool:
        """Check structural equality."""
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
                return t1.left == t2.left and t1.right == t2.right
            return (self._terms_equal(t1.left, t2.left) and
                   self._terms_equal(t1.right, t2.right))

    def _has_redex(self, term: Term) -> bool:
        """Check if term has any redex."""
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

    def compare_reductions(self, term_id: int, term: Term) -> PerformanceComparison:
        """Perform detailed comparison of model vs gold reduction."""
        initial_chars = self.analyze_term_characteristics(term)
        initial_str = Renderer.to_debruijn_with_spans(term).string

        # Run model reduction
        model_trace = self.reduce_with_model_detailed(term, initial_chars)

        # Optionally run gold for comparison
        model_only = self.config.get('model_only', False)
        if model_only:
            # Skip gold comparison - create a dummy gold trace
            gold_trace = None
        else:
            gold_trace = self.reduce_with_gold_detailed(term, initial_chars)

        # Analyze paths
        model_paths = self.analyze_path_patterns(model_trace)
        gold_paths = self.analyze_path_patterns(gold_trace) if gold_trace else None

        if model_only:
            # Model-only mode: just check if model reached normal form
            model_is_nf = not self._has_redex(model_trace.final_term) if model_trace.final_term else True

            return PerformanceComparison(
                term_id=term_id,
                initial_term=initial_str,
                initial_characteristics=initial_chars,
                model_trace=model_trace,
                gold_trace=None,
                same_normal_form=model_is_nf,  # Assume correct if NF
                model_correct=model_is_nf,
                model_faster_steps=True,  # No comparison
                step_difference=0,
                token_difference=0,
                time_difference_ms=model_trace.total_inference_time_ms,
                diverged=False,
                divergence_step=None,
                divergence_type=None,
                model_path_analysis=model_paths,
                gold_path_analysis=None
            )

        # Full comparison mode
        # Check correctness
        model_is_nf = not self._has_redex(model_trace.final_term) if model_trace.final_term else True
        gold_is_nf = not self._has_redex(gold_trace.final_term) if gold_trace.final_term else True

        same_nf = False
        if model_trace.final_term and gold_trace.final_term and model_is_nf and gold_is_nf:
            same_nf = self._terms_equal(model_trace.final_term, gold_trace.final_term)

        # Divergence analysis
        diverged = False
        divergence_step = None
        divergence_type = None

        min_steps = min(len(model_trace.step_metrics), len(gold_trace.step_metrics))
        for i in range(min_steps):
            if model_trace.step_metrics[i].term_string != gold_trace.step_metrics[i].term_string:
                diverged = True
                divergence_step = i
                # Determine divergence type
                if model_trace.step_metrics[i].redex_path != gold_trace.step_metrics[i].redex_path:
                    divergence_type = 'path_choice'
                else:
                    divergence_type = 'unknown'
                break

        if not diverged and len(model_trace.step_metrics) < len(gold_trace.step_metrics):
            diverged = True
            divergence_step = len(model_trace.step_metrics)
            divergence_type = 'early_stop'

        # Compute differences
        step_diff = model_trace.total_steps - gold_trace.total_steps
        token_diff = model_trace.total_tokens - gold_trace.total_tokens
        time_diff = model_trace.total_inference_time_ms

        return PerformanceComparison(
            term_id=term_id,
            initial_term=initial_str,
            initial_characteristics=initial_chars,
            model_trace=model_trace,
            gold_trace=gold_trace,
            same_normal_form=same_nf,
            model_correct=same_nf,
            model_faster_steps=step_diff < 0,
            step_difference=step_diff,
            token_difference=token_diff,
            time_difference_ms=time_diff,
            diverged=diverged,
            divergence_step=divergence_step,
            divergence_type=divergence_type,
            model_path_analysis=model_paths,
            gold_path_analysis=gold_paths
        )

    def run_analysis(self) -> AggregatedMetrics:
        """Run comprehensive performance analysis."""
        num_terms = self.config.get('num_terms', 100)
        verbose = self.config.get('verbose', False)

        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE PERFORMANCE METRICS ANALYSIS")
        print(f"{'='*80}\n")
        print(f"Terms to analyze: {num_terms}")
        print(f"Max depth: {self.config.get('max_depth', 10)}")
        print(f"Max size: {self.config.get('max_size', 100)}")
        print(f"Max steps: {self.config.get('max_steps', 1000)}\n")

        # Generate terms
        print("Generating terms...")
        terms = []
        attempts = 0
        max_attempts = num_terms * 10

        while len(terms) < num_terms and attempts < max_attempts:
            term = self.term_gen.generate()
            if term:
                terms.append(term)
                if len(terms) % 50 == 0:
                    print(f"  Generated {len(terms)}/{num_terms}")
            attempts += 1

        print(f"Generated {len(terms)} terms\n")
        print("Running performance analysis...\n")

        # Analyze each term with timing
        analysis_start = time.perf_counter()
        for i, term in enumerate(terms):
            term_start = time.perf_counter()

            if not verbose and (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - analysis_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(terms) - i - 1)
                print(f"  Analyzed {i + 1}/{len(terms)} terms... "
                      f"(avg: {avg_time:.2f}s/term, est remaining: {remaining:.1f}s)")

            try:
                comparison = self.compare_reductions(i, term)
                self.comparisons.append(comparison)

                term_time = time.perf_counter() - term_start

                if verbose:
                    print(f"\nTerm {i}: {comparison.initial_term[:60]}...")
                    print(f"  Model: {comparison.model_trace.total_steps} steps, "
                          f"{comparison.model_trace.total_inference_time_ms:.2f}ms")
                    if comparison.gold_trace:
                        print(f"  Gold: {comparison.gold_trace.total_steps} steps")
                    print(f"  Correct: {comparison.model_correct}, Diverged: {comparison.diverged}")
                    print(f"  Analysis time: {term_time:.2f}s")
            except Exception as e:
                print(f"  Error on term {i}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue

        print(f"\nCompleted analysis of {len(self.comparisons)} terms\n")

        # Aggregate metrics
        return self.aggregate_metrics()

    def aggregate_metrics(self) -> AggregatedMetrics:
        """Aggregate all metrics across terms."""
        if not self.comparisons:
            raise ValueError("No comparisons to aggregate")

        total = len(self.comparisons)

        # Basic statistics
        model_converged = sum(1 for c in self.comparisons if c.model_trace.converged)
        gold_converged = sum(1 for c in self.comparisons if c.gold_trace and c.gold_trace.converged)
        correct = sum(1 for c in self.comparisons if c.model_correct)
        exact_match = sum(1 for c in self.comparisons if not c.diverged)
        model_faster = sum(1 for c in self.comparisons if c.model_faster_steps)
        diverged = sum(1 for c in self.comparisons if c.diverged)

        # Averages
        avg_step_diff = statistics.mean(c.step_difference for c in self.comparisons)
        avg_token_diff = statistics.mean(c.token_difference for c in self.comparisons)
        avg_inf_time = statistics.mean(c.model_trace.avg_inference_time_ms for c in self.comparisons)

        # Divergence analysis
        div_steps = [c.divergence_step for c in self.comparisons if c.divergence_step is not None]
        avg_div_step = statistics.mean(div_steps) if div_steps else None
        div_types = Counter(c.divergence_type for c in self.comparisons if c.divergence_type)

        # Performance by term characteristics
        def bucket_size(size: int) -> str:
            if size < 20: return 'small'
            elif size < 50: return 'medium'
            elif size < 80: return 'large'
            else: return 'xlarge'

        def bucket_depth(depth: int) -> str:
            if depth < 4: return 'shallow'
            elif depth < 7: return 'medium'
            else: return 'deep'

        def bucket_complexity(comp: float) -> str:
            if comp < 50: return 'simple'
            elif comp < 200: return 'moderate'
            elif comp < 500: return 'complex'
            else: return 'very_complex'

        perf_by_size: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0, 'correct': 0, 'faster': 0, 'avg_step_diff': []
        })
        perf_by_depth: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0, 'correct': 0, 'faster': 0, 'avg_step_diff': []
        })
        perf_by_complexity: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0, 'correct': 0, 'faster': 0, 'avg_step_diff': []
        })

        for c in self.comparisons:
            size_bucket = bucket_size(c.initial_characteristics.size)
            depth_bucket = bucket_depth(c.initial_characteristics.depth)
            comp_bucket = bucket_complexity(c.initial_characteristics.complexity_score)

            for bucket, bucket_name in [
                (perf_by_size, size_bucket),
                (perf_by_depth, depth_bucket),
                (perf_by_complexity, comp_bucket)
            ]:
                bucket[bucket_name]['count'] += 1
                if c.model_correct:
                    bucket[bucket_name]['correct'] += 1
                if c.model_faster_steps:
                    bucket[bucket_name]['faster'] += 1
                bucket[bucket_name]['avg_step_diff'].append(c.step_difference)

        # Convert to percentages
        def finalize_bucket(bucket_dict):
            result = {}
            for k, v in bucket_dict.items():
                count = v['count']
                result[k] = {
                    'count': count,
                    'correct_rate': v['correct'] / count if count > 0 else 0,
                    'faster_rate': v['faster'] / count if count > 0 else 0,
                    'avg_step_diff': statistics.mean(v['avg_step_diff']) if v['avg_step_diff'] else 0
                }
            return result

        perf_by_size_final = finalize_bucket(perf_by_size)
        perf_by_depth_final = finalize_bucket(perf_by_depth)
        perf_by_complexity_final = finalize_bucket(perf_by_complexity)

        # Path pattern analysis
        all_model_patterns = Counter()
        all_gold_patterns = Counter()

        for c in self.comparisons:
            if c.model_path_analysis:
                all_model_patterns.update(c.model_path_analysis.path_patterns)
            if c.gold_path_analysis and c.gold_trace:
                all_gold_patterns.update(c.gold_path_analysis.path_patterns)

        unique_model = [p for p in all_model_patterns if p not in all_gold_patterns]

        # Speed analysis
        speedups = []
        for c in self.comparisons:
            if c.gold_trace and c.gold_trace.total_steps > 0:
                speedup = (c.gold_trace.total_steps - c.model_trace.total_steps) / c.gold_trace.total_steps
                speedups.append((c.term_id, speedup))

        speedups.sort(key=lambda x: -x[1])

        # Fastest/slowest term types
        fastest_terms = speedups[:10]
        slowest_terms = speedups[-10:]

        fastest_types = []
        for term_id, speedup in fastest_terms:
            c = self.comparisons[term_id]
            fastest_types.append({
                'term_id': term_id,
                'speedup': speedup,
                'size': c.initial_characteristics.size,
                'depth': c.initial_characteristics.depth,
                'num_redexes': c.initial_characteristics.num_redexes,
                'complexity': c.initial_characteristics.complexity_score
            })

        slowest_types = []
        for term_id, speedup in slowest_terms:
            c = self.comparisons[term_id]
            slowest_types.append({
                'term_id': term_id,
                'speedup': speedup,
                'size': c.initial_characteristics.size,
                'depth': c.initial_characteristics.depth,
                'num_redexes': c.initial_characteristics.num_redexes,
                'complexity': c.initial_characteristics.complexity_score
            })

        # Error analysis
        error_cases = []
        for c in self.comparisons:
            if not c.model_correct:
                error_cases.append({
                    'term_id': c.term_id,
                    'term': c.initial_term[:100],
                    'size': c.initial_characteristics.size,
                    'depth': c.initial_characteristics.depth,
                    'divergence_step': c.divergence_step,
                    'divergence_type': c.divergence_type
                })

        # Common failure patterns
        failure_patterns = Counter()
        for c in self.comparisons:
            if not c.model_correct:
                size_bucket = bucket_size(c.initial_characteristics.size)
                depth_bucket = bucket_depth(c.initial_characteristics.depth)
                pattern = f"{size_bucket}_{depth_bucket}_{c.divergence_type}"
                failure_patterns[pattern] += 1

        return AggregatedMetrics(
            total_terms=total,
            successful_terms=model_converged,
            model_convergence_rate=model_converged / total,
            gold_convergence_rate=gold_converged / total,
            accuracy=correct / total,
            same_path_rate=exact_match / total,
            model_faster_rate=model_faster / total,
            avg_step_difference=avg_step_diff,
            avg_token_difference=avg_token_diff,
            avg_time_per_inference_ms=avg_inf_time,
            divergence_rate=diverged / total,
            avg_divergence_step=avg_div_step,
            divergence_type_distribution=div_types,
            performance_by_size=perf_by_size_final,
            performance_by_depth=perf_by_depth_final,
            performance_by_complexity=perf_by_complexity_final,
            model_path_patterns=all_model_patterns,
            gold_path_patterns=all_gold_patterns,
            unique_model_patterns=unique_model,
            terms_by_speedup=speedups,
            fastest_term_types=fastest_types,
            slowest_term_types=slowest_types,
            error_cases=error_cases,
            common_failure_patterns=failure_patterns
        )

    def print_report(self, metrics: AggregatedMetrics):
        """Print comprehensive performance report."""
        print(f"\n{'='*80}")
        print(f"PERFORMANCE METRICS REPORT")
        print(f"{'='*80}\n")

        # Overall statistics
        print(f"OVERALL STATISTICS")
        print(f"{'-'*80}")
        print(f"Total terms analyzed: {metrics.total_terms}")
        print(f"Successful reductions: {metrics.successful_terms} ({100*metrics.model_convergence_rate:.1f}%)")
        print(f"Accuracy (correct NF): {100*metrics.accuracy:.1f}%")
        print(f"Same path rate: {100*metrics.same_path_rate:.1f}%")
        print(f"Model faster rate: {100*metrics.model_faster_rate:.1f}%")
        print(f"Average step difference: {metrics.avg_step_difference:+.2f}")
        print(f"Average inference time: {metrics.avg_time_per_inference_ms:.2f}ms")

        # Divergence
        print(f"\nDIVERGENCE ANALYSIS")
        print(f"{'-'*80}")
        print(f"Divergence rate: {100*metrics.divergence_rate:.1f}%")
        if metrics.avg_divergence_step:
            print(f"Average divergence step: {metrics.avg_divergence_step:.1f}")
        print(f"Divergence types:")
        for div_type, count in metrics.divergence_type_distribution.most_common():
            print(f"  {div_type}: {count} ({100*count/metrics.total_terms:.1f}%)")

        # Performance by term size
        print(f"\nPERFORMANCE BY TERM SIZE")
        print(f"{'-'*80}")
        for size_cat in ['small', 'medium', 'large', 'xlarge']:
            if size_cat in metrics.performance_by_size:
                stats = metrics.performance_by_size[size_cat]
                print(f"{size_cat.upper()}: {stats['count']} terms")
                print(f"  Correct: {100*stats['correct_rate']:.1f}%")
                print(f"  Faster: {100*stats['faster_rate']:.1f}%")
                print(f"  Avg step diff: {stats['avg_step_diff']:+.2f}")

        # Performance by depth
        print(f"\nPERFORMANCE BY TERM DEPTH")
        print(f"{'-'*80}")
        for depth_cat in ['shallow', 'medium', 'deep']:
            if depth_cat in metrics.performance_by_depth:
                stats = metrics.performance_by_depth[depth_cat]
                print(f"{depth_cat.upper()}: {stats['count']} terms")
                print(f"  Correct: {100*stats['correct_rate']:.1f}%")
                print(f"  Faster: {100*stats['faster_rate']:.1f}%")
                print(f"  Avg step diff: {stats['avg_step_diff']:+.2f}")

        # Performance by complexity
        print(f"\nPERFORMANCE BY COMPLEXITY")
        print(f"{'-'*80}")
        for comp_cat in ['simple', 'moderate', 'complex', 'very_complex']:
            if comp_cat in metrics.performance_by_complexity:
                stats = metrics.performance_by_complexity[comp_cat]
                print(f"{comp_cat.upper()}: {stats['count']} terms")
                print(f"  Correct: {100*stats['correct_rate']:.1f}%")
                print(f"  Faster: {100*stats['faster_rate']:.1f}%")
                print(f"  Avg step diff: {stats['avg_step_diff']:+.2f}")

        # Path patterns
        print(f"\nPATH PATTERN ANALYSIS")
        print(f"{'-'*80}")
        print("Top model path patterns:")
        for pattern, count in metrics.model_path_patterns.most_common(10):
            print(f"  {pattern}: {count}")

        print("\nTop gold path patterns:")
        for pattern, count in metrics.gold_path_patterns.most_common(10):
            print(f"  {pattern}: {count}")

        if metrics.unique_model_patterns:
            print(f"\nUnique model patterns (not in gold):")
            for pattern in metrics.unique_model_patterns[:5]:
                print(f"  {pattern}")

        # Speed analysis
        print(f"\nSPEED ANALYSIS")
        print(f"{'-'*80}")
        print("Fastest reductions (model vs gold):")
        for item in metrics.fastest_term_types[:5]:
            print(f"  Term {item['term_id']}: {100*item['speedup']:.1f}% speedup")
            print(f"    Size: {item['size']}, Depth: {item['depth']}, Redexes: {item['num_redexes']}")

        print("\nSlowest reductions:")
        for item in metrics.slowest_term_types[:5]:
            print(f"  Term {item['term_id']}: {100*item['speedup']:.1f}% speedup")
            print(f"    Size: {item['size']}, Depth: {item['depth']}, Redexes: {item['num_redexes']}")

        # Error analysis
        if metrics.error_cases:
            print(f"\nERROR ANALYSIS")
            print(f"{'-'*80}")
            print(f"Total errors: {len(metrics.error_cases)}")
            print("Common failure patterns:")
            for pattern, count in metrics.common_failure_patterns.most_common(5):
                print(f"  {pattern}: {count}")

            print("\nSample error cases:")
            for case in metrics.error_cases[:3]:
                print(f"  Term {case['term_id']}: size={case['size']}, depth={case['depth']}")
                print(f"    Diverged at step {case['divergence_step']} ({case['divergence_type']})")
                print(f"    Term: {case['term'][:70]}...")

        print(f"\n{'='*80}\n")

    def save_detailed_results(self, metrics: AggregatedMetrics, output_dir: str):
        """Save comprehensive results to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary metrics
        summary = {
            'overall': {
                'total_terms': metrics.total_terms,
                'successful_terms': metrics.successful_terms,
                'accuracy': metrics.accuracy,
                'same_path_rate': metrics.same_path_rate,
                'model_faster_rate': metrics.model_faster_rate,
                'avg_step_difference': metrics.avg_step_difference,
                'avg_token_difference': metrics.avg_token_difference,
                'avg_time_per_inference_ms': metrics.avg_time_per_inference_ms,
                'divergence_rate': metrics.divergence_rate,
                'avg_divergence_step': metrics.avg_divergence_step,
            },
            'divergence_types': dict(metrics.divergence_type_distribution),
            'performance_by_size': metrics.performance_by_size,
            'performance_by_depth': metrics.performance_by_depth,
            'performance_by_complexity': metrics.performance_by_complexity,
            'path_patterns': {
                'model': dict(metrics.model_path_patterns.most_common(20)),
                'gold': dict(metrics.gold_path_patterns.most_common(20)),
                'unique_model': metrics.unique_model_patterns[:10]
            },
            'speed_analysis': {
                'fastest_terms': metrics.fastest_term_types,
                'slowest_terms': metrics.slowest_term_types
            },
            'errors': {
                'total_errors': len(metrics.error_cases),
                'error_cases': metrics.error_cases[:20],
                'common_patterns': dict(metrics.common_failure_patterns.most_common(10))
            }
        }

        with open(output_path / 'summary_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary metrics saved to: {output_path / 'summary_metrics.json'}")

        # Save detailed comparisons
        detailed = []
        for comp in self.comparisons:
            detailed.append({
                'term_id': comp.term_id,
                'initial_term': comp.initial_term,
                'characteristics': asdict(comp.initial_characteristics),
                'model': {
                    'steps': comp.model_trace.total_steps,
                    'tokens': comp.model_trace.total_tokens,
                    'time_ms': comp.model_trace.total_inference_time_ms,
                    'avg_time_ms': comp.model_trace.avg_inference_time_ms,
                    'converged': comp.model_trace.converged,
                    'size_evolution': comp.model_trace.size_evolution,
                    'depth_evolution': comp.model_trace.depth_evolution,
                },
                'gold': {
                    'steps': comp.gold_trace.total_steps if comp.gold_trace else None,
                    'tokens': comp.gold_trace.total_tokens if comp.gold_trace else None,
                    'converged': comp.gold_trace.converged if comp.gold_trace else None,
                    'size_evolution': comp.gold_trace.size_evolution if comp.gold_trace else None,
                    'depth_evolution': comp.gold_trace.depth_evolution if comp.gold_trace else None,
                } if comp.gold_trace else None,
                'comparison': {
                    'same_normal_form': comp.same_normal_form,
                    'model_correct': comp.model_correct,
                    'model_faster_steps': comp.model_faster_steps,
                    'step_difference': comp.step_difference,
                    'token_difference': comp.token_difference,
                    'time_difference_ms': comp.time_difference_ms,
                    'diverged': comp.diverged,
                    'divergence_step': comp.divergence_step,
                    'divergence_type': comp.divergence_type,
                }
            })

        with open(output_path / 'detailed_comparisons.json', 'w') as f:
            json.dump(detailed, f, indent=2)

        print(f"Detailed comparisons saved to: {output_path / 'detailed_comparisons.json'}")

        # Save step-by-step traces for a sample of interesting cases
        sample_traces = []

        # Add a few fastest cases
        fastest_ids = [t[0] for t in metrics.terms_by_speedup[:3]]
        # Add a few slowest cases
        slowest_ids = [t[0] for t in metrics.terms_by_speedup[-3:]]
        # Add error cases
        error_ids = [e['term_id'] for e in metrics.error_cases[:3]]

        interesting_ids = set(fastest_ids + slowest_ids + error_ids)

        for comp in self.comparisons:
            if comp.term_id in interesting_ids:
                model_steps = []
                for step in comp.model_trace.step_metrics:
                    model_steps.append({
                        'step': step.step_num,
                        'term': step.term_string,
                        'size': step.term_size,
                        'depth': step.term_depth,
                        'redex_path': step.redex_path,
                        'redex_depth': step.redex_depth,
                        'inference_time_ms': step.inference_time_ms,
                        'nf_confidence': step.nf_confidence,
                    })

                gold_steps = []
                if comp.gold_trace:
                    for step in comp.gold_trace.step_metrics:
                        gold_steps.append({
                            'step': step.step_num,
                            'term': step.term_string,
                            'size': step.term_size,
                            'depth': step.term_depth,
                            'redex_path': step.redex_path,
                            'redex_depth': step.redex_depth,
                        })

                sample_traces.append({
                    'term_id': comp.term_id,
                    'initial_term': comp.initial_term,
                    'model_steps': model_steps,
                    'gold_steps': gold_steps if comp.gold_trace else None,
                })

        with open(output_path / 'sample_traces.json', 'w') as f:
            json.dump(sample_traces, f, indent=2)

        print(f"Sample traces saved to: {output_path / 'sample_traces.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive performance metrics for lambda calculus model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-terms', type=int, default=200,
                       help='Number of terms to analyze')
    parser.add_argument('--min-depth', type=int, default=2,
                       help='Minimum term depth')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum term depth')
    parser.add_argument('--max-size', type=int, default=100,
                       help='Maximum term size')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum reduction steps')
    parser.add_argument('--wall-clock-limit-ms', type=float, default=50.0,
                       help='Wall clock limit per term for gold reducer (ms)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed output')
    parser.add_argument('--model-only', action='store_true',
                       help='Skip gold comparison for faster analysis')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')

    args = parser.parse_args()
    config = vars(args)

    # Run analysis
    analyzer = PerformanceAnalyzer(args.checkpoint, config)
    metrics = analyzer.run_analysis()

    # Print report
    analyzer.print_report(metrics)

    # Save results
    analyzer.save_detailed_results(metrics, args.output_dir)

    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
