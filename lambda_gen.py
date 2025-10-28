#!/usr/bin/env python3
#
#Lambda Calculus Training Data Generator for 700M-class Models
#==============================================================

#Generates high-SNR λ-calculus training data for learned β-reduction.

#TOKEN DESIGN:
#  - Minimal ASCII: \ for lambda, . for dot, () for grouping
#  - Variables: single lowercase a-z, then a0-a9, etc. (named) or digits 0-9 (de Bruijn)
#  - No spaces inside terms; juxtaposition for application
#  - Both renderers emit dot after binder: \. in de Bruijn, \x. in named

#STRATEGIES:
#  - normal: leftmost-outermost tree β-reduction
#  - levy_like: call-by-need graph reduction with thunk memoization

#SUPERVISION TARGETS:
#  - Span-only (default): target_span points to next redex. Lightweight, high throughput.
#  - Span + next term (--emit-next): adds next_term field for supervised generation.
#  - Full trace (--emit-trace): stream entire reduction sequence with trace_id.
#  - Choose based on model capacity and compute budget.

#SCHEMA (JSONL per line):
#  {
#    "strategy": str,
#    "render": "debruijn" | "named",
#    "term": str,
#    "step_k": int,
#    "target_span": [int, int],
#    "next_term": str (optional),
#    "normal_form": str (optional),
#    "steps_total": int,
#    "diverged": bool,
#    "trace_id": str (optional),
#    "meta": {size, depth, libs, seed, draw_index, uid, thunk_evals, thunk_hits,
#             schema_version, term_hash, max_steps, fuel_remaining, fuel_consumed_ratio,
#             is_pathological, size_growth_rate}
#  }

#USAGE:
#  Live:     python generator.py live --strategy levy_like --share --rate 100
#  Test:     python generator.py test --n 20 --show-traces --seed 1337
#  Validate: python generator.py validate --n 2000 --max-depth 10

#SHARING METRICS (levy_like with --share):
#  thunk_evals: number of thunks forced for first time
#  thunk_hits: number of cache hits on already-evaluated thunks
#  share_hit_rate: hits / (hits + evals), measures duplication avoided
#

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Iterator
from enum import Enum

SCHEMA_VERSION = "2.0"

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ============================================================================
# TREE TERMS (for normal-order)
# ============================================================================

class TermType(Enum):
    VAR = 1
    ABS = 2
    APP = 3

@dataclass(slots=True, frozen=True)
class Term:
    #Immutable tree lambda term.
    type: TermType
    var: Optional[int] = field(default=None)
    body: Optional['Term'] = field(default=None)
    left: Optional['Term'] = field(default=None)
    right: Optional['Term'] = field(default=None)
    
    def size(self) -> int:
        if self.type == TermType.VAR:
            return 1
        elif self.type == TermType.ABS:
            return 1 + self.body.size()
        else:
            return 1 + self.left.size() + self.right.size()
    
    def depth(self) -> int:
        if self.type == TermType.VAR:
            return 0
        elif self.type == TermType.ABS:
            return 1 + self.body.depth()
        else:
            return 1 + max(self.left.depth(), self.right.depth())

# ============================================================================
# GRAPH TERMS (for call-by-need)
# ============================================================================

class NodeKind(Enum):
    VAR = 1
    ABS = 2
    APP = 3
    THUNK = 4
    VALUE = 5

@dataclass(slots=True)
class GraphNode:
    #DAG node for call-by-need reduction with sharing.
    kind: NodeKind
    var: Optional[int] = field(default=None)
    body: Optional['GraphNode'] = field(default=None)
    left: Optional['GraphNode'] = field(default=None)
    right: Optional['GraphNode'] = field(default=None)
    env: Optional[List['GraphNode']] = field(default=None)
    evaluated: bool = field(default=False)
    cache: Optional['GraphNode'] = field(default=None)
    
    def to_tree(self, visited: Optional[set] = None) -> Term:
        #Project graph node back to tree term for rendering with cycle detection.

        #Note: VAR nodes with THUNK bindings project the variable reference itself
        #to avoid premature expansion in trace views. The redex selection logic
        #operates on the projected tree, so this preserves correctness.
        #
        if visited is None:
            visited = set()

        node_id = id(self)
        if node_id in visited:
            # Cycle detected - return a safe placeholder
            return Term(TermType.VAR, var=0)
        visited.add(node_id)

        if self.kind == NodeKind.VALUE:
            result = self.cache.to_tree(visited) if self.cache else Term(TermType.VAR, var=0)
        elif self.kind == NodeKind.THUNK:
            # Check external cache status via node fields (kept for backward compat)
            if self.evaluated and self.cache:
                result = self.cache.to_tree(visited)
            elif self.body:
                result = self.body.to_tree(visited)
            else:
                result = Term(TermType.VAR, var=0)
        elif self.kind == NodeKind.VAR:
            if self.env and self.var is not None and self.var < len(self.env):
                binding = self.env[self.var]
                if binding.kind == NodeKind.THUNK and not binding.evaluated:
                    result = Term(TermType.VAR, var=self.var)
                else:
                    result = binding.to_tree(visited)
            else:
                result = Term(TermType.VAR, var=self.var if self.var is not None else 0)
        elif self.kind == NodeKind.ABS:
            result = Term(TermType.ABS, body=self.body.to_tree(visited) if self.body else Term(TermType.VAR, var=0))
        else:  # APP
            result = Term(TermType.APP,
                       left=self.left.to_tree(visited) if self.left else Term(TermType.VAR, var=0),
                       right=self.right.to_tree(visited) if self.right else Term(TermType.VAR, var=0))

        visited.discard(node_id)
        return result

# ============================================================================
# TERM GENERATION
# ============================================================================

class TermGenerator:
    #Generates random well-formed lambda terms.#
    
    def __init__(self, rng: random.Random, max_depth: int, min_depth: int, 
                 max_size: int, libraries: List[str], allow_divergent: bool):
        self.rng = rng
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.max_size = max_size
        self.libraries = libraries
        self.allow_divergent = allow_divergent
    
    def generate(self) -> Optional[Term]:
        #Generate a random term with size/depth distribution sampling for better coverage.#
        # Sample target size and depth from distributions to ensure good coverage
        # Use log-uniform for sizes to get better representation across the range
        # Use uniform for depths within valid range

        for attempt in range(100):
            # 70% of the time: sample size/depth from distribution for diversity
            # 30% of the time: use full range for exploration
            if self.rng.random() < 0.7 and attempt < 80:
                # Log-uniform sampling for size (more small and medium terms)
                log_min = 1
                log_max = max(2, int(self.max_size ** 0.5))
                target_size = min(self.max_size, self.rng.randint(log_min, log_max) ** 2)

                # Uniform sampling for depth
                target_depth = self.rng.randint(self.min_depth, self.max_depth)

                term = self._gen_term(0, 0)
                if term and term.depth() >= self.min_depth and term.size() <= self.max_size:
                    # Accept if close to target (within 40%) or if we've tried many times
                    size_ok = attempt > 40 or abs(term.size() - target_size) <= max(5, target_size * 0.4)
                    depth_ok = attempt > 40 or abs(term.depth() - target_depth) <= 3
                    if size_ok and depth_ok:
                        return term
            else:
                # Full range exploration - just accept any valid term
                term = self._gen_term(0, 0)
                if term and term.depth() >= self.min_depth and term.size() <= self.max_size:
                    return term

        return None
    
    def _gen_term(self, depth: int, bound_vars: int) -> Optional[Term]:
        #Recursively generate term.#
        if depth >= self.max_depth:
            if bound_vars > 0:
                return Term(TermType.VAR, var=self.rng.randrange(bound_vars))
            return None
        
        weights = [0.3, 0.3, 0.4] if depth < self.max_depth - 1 else [1.0, 0.0, 0.0]
        choice = self.rng.choices(['var', 'abs', 'app'], weights=weights)[0]
        
        if choice == 'var':
            if bound_vars == 0:
                choice = 'abs'
            else:
                return Term(TermType.VAR, var=self.rng.randrange(bound_vars))
        
        if choice == 'abs':
            body = self._gen_term(depth + 1, bound_vars + 1)
            if body:
                return Term(TermType.ABS, body=body)
            return None
        
        left = self._gen_term(depth + 1, bound_vars)
        right = self._gen_term(depth + 1, bound_vars)
        if left and right:
            return Term(TermType.APP, left=left, right=right)
        return None
    
    def library_term(self, name: str) -> Optional[Term]:
        #Generate standard combinator/library terms.#
        if name == 'church':
            n = self.rng.randint(0, 5)
            body = Term(TermType.VAR, var=0)
            for _ in range(n):
                body = Term(TermType.APP,
                          left=Term(TermType.VAR, var=1),
                          right=body)
            inner = Term(TermType.ABS, body=body)
            return Term(TermType.ABS, body=inner)

        elif name == 'ski':
            choice = self.rng.choice(['S', 'K', 'I'])
            if choice == 'I':
                # I = \x.x
                return Term(TermType.ABS, body=Term(TermType.VAR, var=0))
            elif choice == 'K':
                # K = \x.\y.x
                inner = Term(TermType.ABS, body=Term(TermType.VAR, var=1))
                return Term(TermType.ABS, body=inner)
            else:
                # S = \x.\y.\z.xz(yz)
                z = Term(TermType.VAR, var=0)
                y = Term(TermType.VAR, var=1)
                x = Term(TermType.VAR, var=2)
                yz = Term(TermType.APP, left=y, right=z)
                xz = Term(TermType.APP, left=x, right=z)
                body = Term(TermType.APP, left=xz, right=yz)
                l3 = Term(TermType.ABS, body=body)
                l2 = Term(TermType.ABS, body=l3)
                return Term(TermType.ABS, body=l2)

        elif name == 'combinators':
            # Extended combinator library: B, C, W, etc.
            choice = self.rng.choice(['B', 'C', 'W'])
            if choice == 'B':
                # B = \x.\y.\z.x(yz) (composition)
                z = Term(TermType.VAR, var=0)
                y = Term(TermType.VAR, var=1)
                x = Term(TermType.VAR, var=2)
                yz = Term(TermType.APP, left=y, right=z)
                body = Term(TermType.APP, left=x, right=yz)
                l3 = Term(TermType.ABS, body=body)
                l2 = Term(TermType.ABS, body=l3)
                return Term(TermType.ABS, body=l2)
            elif choice == 'C':
                # C = \x.\y.\z.xzy (flip)
                z = Term(TermType.VAR, var=0)
                y = Term(TermType.VAR, var=1)
                x = Term(TermType.VAR, var=2)
                xz = Term(TermType.APP, left=x, right=z)
                body = Term(TermType.APP, left=xz, right=y)
                l3 = Term(TermType.ABS, body=body)
                l2 = Term(TermType.ABS, body=l3)
                return Term(TermType.ABS, body=l2)
            else:  # W
                # W = \x.\y.xyy (duplication)
                y = Term(TermType.VAR, var=0)
                x = Term(TermType.VAR, var=1)
                xy = Term(TermType.APP, left=x, right=y)
                body = Term(TermType.APP, left=xy, right=y)
                l2 = Term(TermType.ABS, body=body)
                return Term(TermType.ABS, body=l2)

        elif name == 'y':
            if not self.allow_divergent:
                return None
            # Y = \f.(\x.f(xx))(\x.f(xx))
            xx = Term(TermType.APP,
                     left=Term(TermType.VAR, var=0),
                     right=Term(TermType.VAR, var=0))
            fxx = Term(TermType.APP,
                      left=Term(TermType.VAR, var=1),
                      right=xx)
            inner = Term(TermType.ABS, body=fxx)
            app = Term(TermType.APP, left=inner, right=inner)
            return Term(TermType.ABS, body=app)

        elif name == 'omega':
            if not self.allow_divergent:
                return None
            # ω = \x.xx (self-application)
            xx = Term(TermType.APP,
                     left=Term(TermType.VAR, var=0),
                     right=Term(TermType.VAR, var=0))
            return Term(TermType.ABS, body=xx)

        elif name == 'booleans':
            choice = self.rng.choice(['true', 'false'])
            if choice == 'true':
                # True = \x.\y.x
                inner = Term(TermType.ABS, body=Term(TermType.VAR, var=1))
                return Term(TermType.ABS, body=inner)
            else:
                # False = \x.\y.y
                inner = Term(TermType.ABS, body=Term(TermType.VAR, var=0))
                return Term(TermType.ABS, body=inner)

        elif name == 'pairs':
            # Pair encoding and projection functions
            choice = self.rng.choice(['pair', 'fst', 'snd'])
            if choice == 'pair':
                # pair = \x.\y.\z.zxy
                z = Term(TermType.VAR, var=0)
                y = Term(TermType.VAR, var=1)
                x = Term(TermType.VAR, var=2)
                zx = Term(TermType.APP, left=z, right=x)
                body = Term(TermType.APP, left=zx, right=y)
                l3 = Term(TermType.ABS, body=body)
                l2 = Term(TermType.ABS, body=l3)
                return Term(TermType.ABS, body=l2)
            elif choice == 'fst':
                # fst = \p.p(\x.\y.x) = \p.p(True)
                y = Term(TermType.VAR, var=0)
                x = Term(TermType.VAR, var=1)
                inner = Term(TermType.ABS, body=x)
                l2 = Term(TermType.ABS, body=inner)
                p = Term(TermType.VAR, var=0)
                body = Term(TermType.APP, left=p, right=l2)
                return Term(TermType.ABS, body=body)
            else:  # snd
                # snd = \p.p(\x.\y.y) = \p.p(False)
                y = Term(TermType.VAR, var=0)
                x = Term(TermType.VAR, var=1)
                inner = Term(TermType.ABS, body=y)
                l2 = Term(TermType.ABS, body=inner)
                p = Term(TermType.VAR, var=0)
                body = Term(TermType.APP, left=p, right=l2)
                return Term(TermType.ABS, body=body)

        return None

# ============================================================================
# RENDERING WITH SPAN TRACKING
# ============================================================================

@dataclass
class RenderResult:
    #Result of rendering with span tracking.#
    string: str
    spans: Dict[int, Tuple[int, int]]

class Renderer:
    #Renders terms to minimal token strings with span tracking.
    
    #Parenthesis discipline: only add parens when left child is ABS or right child is ABS.
    #This exploits left-associative application to minimize token count.
    #
    
    @staticmethod
    def to_debruijn_with_spans(term: Term) -> RenderResult:
        #Render with de Bruijn indices, tracking node spans.#
        spans = {}
        parts = []
        pos = [0]
        
        def render(t: Term, node_id: int):
            start = pos[0]
            
            if t.type == TermType.VAR:
                s = str(t.var)
                parts.append(s)
                pos[0] += len(s)
            elif t.type == TermType.ABS:
                parts.append('\\.')
                pos[0] += 2
                render(t.body, node_id * 2 + 1)
            else:
                need_left_paren = t.left.type == TermType.ABS
                need_right_paren = t.right.type == TermType.ABS
                
                if need_left_paren:
                    parts.append('(')
                    pos[0] += 1
                render(t.left, node_id * 2 + 1)
                if need_left_paren:
                    parts.append(')')
                    pos[0] += 1
                
                if need_right_paren:
                    parts.append('(')
                    pos[0] += 1
                render(t.right, node_id * 2 + 2)
                if need_right_paren:
                    parts.append(')')
                    pos[0] += 1
            
            spans[node_id] = (start, pos[0])
        
        render(term, 1)
        return RenderResult(''.join(parts), spans)
    
    @staticmethod
    def to_named_with_spans(term: Term) -> RenderResult:
        #Render with named variables, tracking node spans.#
        spans = {}
        parts = []
        pos = [0]
        
        def var_name(index: int) -> str:
            if index < 26:
                return chr(ord('a') + index)
            else:
                return chr(ord('a') + (index % 26)) + str(index // 26)
        
        def render(t: Term, env: List[str], node_id: int):
            start = pos[0]
            
            if t.type == TermType.VAR:
                s = env[t.var] if t.var < len(env) else f"?{t.var}"
                parts.append(s)
                pos[0] += len(s)
            elif t.type == TermType.ABS:
                vname = var_name(len(env))
                parts.append(f'\\{vname}.')
                pos[0] += len(vname) + 2
                new_env = [vname] + env
                render(t.body, new_env, node_id * 2 + 1)
            else:
                need_left_paren = t.left.type == TermType.ABS
                need_right_paren = t.right.type == TermType.ABS
                
                if need_left_paren:
                    parts.append('(')
                    pos[0] += 1
                render(t.left, env, node_id * 2 + 1)
                if need_left_paren:
                    parts.append(')')
                    pos[0] += 1
                
                if need_right_paren:
                    parts.append('(')
                    pos[0] += 1
                render(t.right, env, node_id * 2 + 2)
                if need_right_paren:
                    parts.append(')')
                    pos[0] += 1
            
            spans[node_id] = (start, pos[0])
        
        render(term, [], 1)
        return RenderResult(''.join(parts), spans)
    
    @staticmethod
    def tokenize(s: str) -> List[str]:
        #Split rendered string into tokens.#
        tokens = []
        i = 0
        while i < len(s):
            c = s[i]
            if c in '\\().':
                tokens.append(c)
                i += 1
            elif c.isalnum():
                j = i
                while j < len(s) and s[j].isalnum():
                    j += 1
                tokens.append(s[i:j])
                i = j
            else:
                i += 1
        return tokens

# ============================================================================
# REFERENCE SUBSTITUTION (for validation)
# ============================================================================

def reference_substitute(term: Term, var: int, replacement: Term) -> Term:
    #Slow, obviously-correct capture-avoiding substitution.#
    def shift(t: Term, amount: int, cutoff: int = 0) -> Term:
        if t.type == TermType.VAR:
            if t.var >= cutoff:
                return Term(TermType.VAR, var=t.var + amount)
            return t
        elif t.type == TermType.ABS:
            return Term(TermType.ABS, body=shift(t.body, amount, cutoff + 1))
        else:
            return Term(TermType.APP, left=shift(t.left, amount, cutoff),
                       right=shift(t.right, amount, cutoff))
    
    def subst(t: Term, v: int, repl: Term, depth: int = 0) -> Term:
        if t.type == TermType.VAR:
            if t.var == v + depth:
                return shift(repl, depth)
            return t
        elif t.type == TermType.ABS:
            return Term(TermType.ABS, body=subst(t.body, v, repl, depth + 1))
        else:
            return Term(TermType.APP, left=subst(t.left, v, repl, depth),
                       right=subst(t.right, v, repl, depth))
    
    return subst(term, var, replacement, 0)

def terms_alpha_equiv(t1: Term, t2: Term) -> bool:
    #Check α-equivalence via de Bruijn comparison.#
    r1 = Renderer.to_debruijn_with_spans(t1)
    r2 = Renderer.to_debruijn_with_spans(t2)
    return r1.string == r2.string

# ============================================================================
# TREE REDUCTION (normal-order)
# ============================================================================

class TreeReducer:
    #Normal-order β-reduction on trees.#
    
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
    
    def reduce(self, term: Term) -> Tuple[List[Tuple[Term, Optional[List[int]]]], bool]:
        #Reduce term, return trace with redex paths aligned at each step.#
        trace = []
        current = term

        for step in range(self.max_steps):
            redex_path = self._find_leftmost_outermost(current)
            trace.append((current, redex_path))

            if redex_path is None:
                return trace, False

            current = self._apply_reduction(current, redex_path)

        final_redex = self._find_leftmost_outermost(current)
        trace.append((current, final_redex))
        return trace, True
    
    def _find_leftmost_outermost(self, term: Term, path: Optional[List[int]] = None) -> Optional[List[int]]:
        #Find leftmost-outermost β-redex.#
        if path is None:
            path = []
        
        if term.type == TermType.APP and term.left.type == TermType.ABS:
            return path
        
        if term.type == TermType.ABS:
            return self._find_leftmost_outermost(term.body, path + [0])
        elif term.type == TermType.APP:
            left_result = self._find_leftmost_outermost(term.left, path + [0])
            if left_result:
                return left_result
            return self._find_leftmost_outermost(term.right, path + [1])
        
        return None
    
    def _apply_reduction(self, term: Term, path: List[int]) -> Term:
        #Apply β-reduction at specified path.#
        if not path:
            assert term.type == TermType.APP and term.left.type == TermType.ABS
            return reference_substitute(term.left.body, 0, term.right)
        
        direction = path[0]
        if term.type == TermType.ABS:
            return Term(TermType.ABS, body=self._apply_reduction(term.body, path[1:]))
        else:
            if direction == 0:
                return Term(TermType.APP, left=self._apply_reduction(term.left, path[1:]),
                           right=term.right)
            else:
                return Term(TermType.APP, left=term.left,
                           right=self._apply_reduction(term.right, path[1:]))

# ============================================================================
# GRAPH REDUCTION (call-by-need with sharing)
# ============================================================================

class GraphReducer:
    #Call-by-need reduction with thunk memoization and wall clock limits.

    #Environment semantics: env lists are treated as immutable closures by design.
    #Child nodes receive the same env reference, which is read-only throughout.
    #
    #Wall clock limiting: Reduction aborts if wall_clock_limit_ms is exceeded.
    #This ensures predictable throughput independent of term complexity.
    #The model becomes runtime-aware through step_ms metrics in training data.
    #

    def __init__(self, wall_clock_limit_ms: float = 100.0, max_steps: int = 10000):
        self.wall_clock_limit_ms = wall_clock_limit_ms  # Primary limiter
        self.max_steps = max_steps  # Safety fallback (should rarely hit)
        self.thunk_evals = 0
        self.thunk_hits = 0
        # Track evaluated thunks separately to avoid mutating nodes
        self.thunk_cache: Dict[int, GraphNode] = {}
    
    def term_to_graph(self, term: Term, env: Optional[List[GraphNode]] = None) -> GraphNode:
        #Convert tree term to graph node.#
        if env is None:
            env = []
        
        if term.type == TermType.VAR:
            return GraphNode(NodeKind.VAR, var=term.var, env=env)
        elif term.type == TermType.ABS:
            body = self.term_to_graph(term.body, env)
            return GraphNode(NodeKind.ABS, body=body, env=env)
        else:
            left = self.term_to_graph(term.left, env)
            right = self.term_to_graph(term.right, env)
            return GraphNode(NodeKind.APP, left=left, right=right, env=env)
    
    def reduce(self, term: Term) -> Tuple[List[Tuple[Term, Optional[List[int]], float]], bool, int, int, float]:
        #Reduce with call-by-need, return trace with timing stats.
        #
        #Returns:
        #  trace: List[(term, redex_path, step_time_ms)] - each step with timing
        #  diverged: bool - whether wall clock or step limit was hit
        #  thunk_evals: int - sharing metric
        #  thunk_hits: int - sharing metric
        #  total_time_ms: float - total wall clock time
        #
        graph = self.term_to_graph(term)
        trace = []

        # Track wall clock time
        start_time = time.time()
        last_step_time = start_time

        for step in range(self.max_steps):
            step_start = time.time()

            # Wall clock check BEFORE expensive operations
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.wall_clock_limit_ms:
                # Exceeded wall clock limit - treat as diverged
                total_time_ms = elapsed_ms
                final_tree = graph.to_tree()
                final_redex = self._find_redex(final_tree)
                final_step_ms = (time.time() - step_start) * 1000
                trace.append((final_tree, final_redex, final_step_ms))
                return trace, True, self.thunk_evals, self.thunk_hits, total_time_ms

            tree_proj = graph.to_tree()
            redex_path = self._find_redex(tree_proj)

            step_time_ms = (time.time() - step_start) * 1000
            trace.append((tree_proj, redex_path, step_time_ms))

            # CRITICAL FIX #1: None means NF, [] means root redex
            if redex_path is None:
                total_time_ms = (time.time() - start_time) * 1000
                return trace, False, self.thunk_evals, self.thunk_hits, total_time_ms

            graph = self._reduce_at_path(graph, redex_path)
            last_step_time = time.time()

        # Hit max_steps (safety fallback)
        final_tree = graph.to_tree()
        final_redex = self._find_redex(final_tree)
        final_step_ms = (time.time() - last_step_time) * 1000
        trace.append((final_tree, final_redex, final_step_ms))
        total_time_ms = (time.time() - start_time) * 1000
        return trace, True, self.thunk_evals, self.thunk_hits, total_time_ms
    
    def _find_redex(self, term: Term) -> Optional[List[int]]:
        #Find leftmost-outermost redex in projected tree.#
        def search(t: Term, path: List[int]) -> Optional[List[int]]:
            if t.type == TermType.APP and t.left.type == TermType.ABS:
                return path
            if t.type == TermType.ABS:
                return search(t.body, path + [0])
            elif t.type == TermType.APP:
                left_result = search(t.left, path + [0])
                if left_result:
                    return left_result
                return search(t.right, path + [1])
            return None
        
        return search(term, [])
    
    def _reduce_at_path(self, graph: GraphNode, path: List[int]) -> GraphNode:
        #Apply β-reduction at path with thunk binding.#
        if not path:
            assert graph.kind == NodeKind.APP and graph.left.kind == NodeKind.ABS
            arg_thunk = GraphNode(NodeKind.THUNK, body=graph.right, 
                                env=graph.right.env, evaluated=False)
            new_env = [arg_thunk] + (graph.left.env or [])
            return self._instantiate(graph.left.body, new_env)
        
        direction = path[0]
        if graph.kind == NodeKind.ABS:
            new_body = self._reduce_at_path(graph.body, path[1:])
            return GraphNode(NodeKind.ABS, body=new_body, env=graph.env)
        else:
            if direction == 0:
                new_left = self._reduce_at_path(graph.left, path[1:])
                return GraphNode(NodeKind.APP, left=new_left, right=graph.right, env=graph.env)
            else:
                new_right = self._reduce_at_path(graph.right, path[1:])
                return GraphNode(NodeKind.APP, left=graph.left, right=new_right, env=graph.env)
    
    def _instantiate(self, node: GraphNode, env: List[GraphNode]) -> GraphNode:
        #Instantiate node with new environment.#
        if node.kind == NodeKind.VAR:
            if node.var < len(env):
                return self._force(env[node.var])
            return GraphNode(NodeKind.VAR, var=node.var, env=env)
        elif node.kind == NodeKind.ABS:
            new_body = self._instantiate(node.body, env)
            return GraphNode(NodeKind.ABS, body=new_body, env=env)
        else:
            new_left = self._instantiate(node.left, env)
            new_right = self._instantiate(node.right, env)
            return GraphNode(NodeKind.APP, left=new_left, right=new_right, env=env)
    
    def _force(self, thunk: GraphNode) -> GraphNode:
        #Force a thunk, using external cache to avoid mutating nodes.#
        if thunk.kind != NodeKind.THUNK:
            return thunk

        thunk_id = id(thunk)
        if thunk_id in self.thunk_cache:
            self.thunk_hits += 1
            return self.thunk_cache[thunk_id]

        self.thunk_evals += 1
        value = self._instantiate(thunk.body, thunk.env)
        self.thunk_cache[thunk_id] = value
        return value

# ============================================================================
# SPAN CALCULATION
# ============================================================================

def get_redex_span(term: Term, redex_path: Optional[List[int]], render_mode: str) -> Tuple[int, int]:
    #Calculate token span for redex using renderer span map.#
    if render_mode == 'debruijn':
        result = Renderer.to_debruijn_with_spans(term)
    else:
        result = Renderer.to_named_with_spans(term)

    # None means no redex (normal form), empty list means redex at root
    if redex_path is None:
        return (0, 0)

    node_id = path_to_node_id(redex_path)
    if node_id in result.spans:
        return result.spans[node_id]
    return (0, len(result.string))

def path_to_node_id(path: List[int]) -> int:
    #Convert path to node id.#
    node_id = 1
    for direction in path:
        node_id = node_id * 2 + 1 + direction
    return node_id

def term_hash(term_str: str) -> str:
    #Compute SHA-1 hash of term for cross-run auditing.#
    return hashlib.sha1(term_str.encode('utf-8')).hexdigest()[:16]

# ============================================================================
# EXAMPLE GENERATION
# ============================================================================

@dataclass
class Config:
    strategy: str = 'normal'
    render: str = 'debruijn'
    max_depth: int = 8
    min_depth: int = 2
    max_size: int = 50
    wall_clock_limit_ms: float = 100.0  # Primary limiter: wall clock time
    max_steps: int = 10000  # Safety fallback
    share: bool = False
    libraries: Optional[List[str]] = None
    emit_next: bool = False
    emit_nf: bool = False
    emit_trace: bool = False
    allow_divergent: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        if self.libraries is None:
            self.libraries = []

def generate_example(config: Config, rng: random.Random, draw_index: int) -> Optional[Any]:
    #Generate a single training example.#
    gen = TermGenerator(rng, config.max_depth, config.min_depth, 
                       config.max_size, config.libraries, config.allow_divergent)
    
    # Try to generate a term with limited retries
    term = None
    for attempt in range(10):
        if config.libraries and rng.random() < 0.2:
            lib = rng.choice(config.libraries)
            term = gen.library_term(lib)
            if not term:
                term = gen.generate()
        else:
            term = gen.generate()
        
        if term:
            break
    
    if not term:
        return None
    
    # Reduce based on strategy
    try:
        if config.strategy == 'levy_like' and config.share:
            graph_reducer = GraphReducer(config.wall_clock_limit_ms, config.max_steps)
            trace, diverged, thunk_evals, thunk_hits, total_time_ms = graph_reducer.reduce(term)
        else:
            tree_reducer = TreeReducer(config.max_steps)
            trace, diverged = tree_reducer.reduce(term)
            thunk_evals, thunk_hits, total_time_ms = 0, 0, 0.0
    except Exception as e:
        sys.stderr.write(f"\n[Error during reduction: {e}]\n")
        sys.stderr.flush()
        return None
    
    if diverged and not config.allow_divergent:
        return None
    
    if not trace or len(trace) == 0:
        return None
    
    # ALWAYS emit full traces for sequential training
    # Random sampling (lines 774-817) was causing distribution mismatch
    # Model needs to see ALL steps to learn proper reduction sequences
    examples = []
    trace_id = str(uuid.uuid4())

    # Compute initial term size for growth tracking
    # Handle both old format (term, path) and new format (term, path, step_ms)
    if len(trace[0]) == 3:
        initial_term, _, _ = trace[0]
        has_timing = True
    else:
        initial_term, _ = trace[0]
        has_timing = False
    initial_size = initial_term.size()

    # Compute timing stats if available
    if has_timing:
        step_times = [step_ms for _, _, step_ms in trace]
        avg_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
    else:
        step_times = [0.0] * len(trace)
        avg_step_ms = 0.0
        total_time_ms = 0.0

    steps_total = len(trace) - 1

    for step_k in range(len(trace)):
        if has_timing:
            current_term, redex_path, step_ms = trace[step_k]
        else:
            current_term, redex_path = trace[step_k]
            step_ms = 0.0

        if config.render == 'debruijn':
            result = Renderer.to_debruijn_with_spans(current_term)
        else:
            result = Renderer.to_named_with_spans(current_term)

        # CRITICAL FIX #2: Pass redex_path verbatim (None or [])
        # Bug was: redex_path or [] converts None (NF) to [] (root), poisoning labels
        target_span = list(get_redex_span(current_term, redex_path, config.render))

        # Wall clock runtime metrics for model awareness
        elapsed_time_ms = sum(step_times[:step_k+1])
        time_remaining_ms = max(0.0, config.wall_clock_limit_ms - elapsed_time_ms)
        time_consumed_ratio = elapsed_time_ms / config.wall_clock_limit_ms if config.wall_clock_limit_ms > 0 else 0.0

        # Detect pathological cases based on runtime
        current_size = current_term.size()
        size_growth_rate = (current_size / initial_size) if initial_size > 0 else 1.0
        is_pathological = (
            time_consumed_ratio > 0.8 or     # Used >80% of wall clock budget
            avg_step_ms > 5.0 or              # Slow steps (>5ms avg)
            size_growth_rate > 3.0 or         # Size tripled
            current_size > 200                # Very large term
        )

        example = {
            'strategy': config.strategy,
            'render': config.render,
            'term': result.string,
            'step_k': step_k,
            'target_span': target_span,
            'steps_total': steps_total,
            'diverged': diverged,
            'trace_id': trace_id,
            'meta': {
                'size': current_size,
                'depth': current_term.depth(),
                'libs': config.libraries,
                'seed': config.seed,
                'draw_index': draw_index,
                'uid': trace_id,
                'thunk_evals': thunk_evals,
                'thunk_hits': thunk_hits,
                'schema_version': SCHEMA_VERSION,
                'term_hash': term_hash(result.string),
                # Runtime metrics (model is runtime-aware)
                'step_ms': step_ms,
                'avg_step_ms': avg_step_ms,
                'total_time_ms': total_time_ms,
                'wall_clock_limit_ms': config.wall_clock_limit_ms,
                'time_remaining_ms': time_remaining_ms,
                'time_consumed_ratio': time_consumed_ratio,
                'is_pathological': is_pathological,
                'size_growth_rate': size_growth_rate,
                'initial_size': initial_size
            }
        }

        if config.emit_next and step_k < len(trace) - 1:
            if has_timing:
                next_term, _, _ = trace[step_k + 1]
            else:
                next_term, _ = trace[step_k + 1]
            next_result = Renderer.to_debruijn_with_spans(next_term) if config.render == 'debruijn' else Renderer.to_named_with_spans(next_term)
            example['next_term'] = next_result.string

        if config.emit_nf and not diverged and step_k == len(trace) - 1:
            example['normal_form'] = result.string

        examples.append(example)

    return examples if examples else None

def yield_examples(config: Config) -> Iterator[Dict[str, Any]]:
    #Generator yielding training examples.#
    rng = random.Random(config.seed)
    draw_index = 0
    consecutive_failures = 0
    max_consecutive_failures = 1000
    
    while True:
        result = generate_example(config, rng, draw_index)
        if result:
            if isinstance(result, list):
                for ex in result:
                    yield ex
            else:
                yield result
            draw_index += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                sys.stderr.write(f"\n[ERROR: {max_consecutive_failures} consecutive generation failures]\n")
                sys.stderr.write("[Params may be too restrictive - try increasing --max-depth or --max-steps]\n")
                sys.stderr.flush()
                raise RuntimeError(f"Failed to generate examples after {max_consecutive_failures} attempts")
            
            if consecutive_failures % 100 == 0:
                sys.stderr.write(f"\n[Warning: {consecutive_failures} consecutive generation failures]\n")
                sys.stderr.flush()

# ============================================================================
# METRICS
# ============================================================================

class Metrics:
    #Track and report generation metrics.#

    def __init__(self):
        self.count = 0
        self.tokens = 0
        self.sizes = deque(maxlen=1000)
        self.depths = deque(maxlen=1000)
        self.steps = deque(maxlen=1000)
        self.latencies = deque(maxlen=512)
        self.diverged = 0
        self.thunk_evals_total = 0
        self.thunk_hits_total = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.recent_count = 0
        # Pathological case tracking
        self.pathological_count = 0
        self.time_consumed_ratios = deque(maxlen=1000)
        self.size_growth_rates = deque(maxlen=1000)
        self.step_times_ms = deque(maxlen=1000)
    
    def update(self, example: Dict[str, Any], latency: float):
        self.count += 1
        self.tokens += len(example['term'])
        self.sizes.append(example['meta']['size'])
        self.depths.append(example['meta']['depth'])
        self.steps.append(example['steps_total'])
        self.latencies.append(latency)
        if example['diverged']:
            self.diverged += 1
        self.thunk_evals_total += example['meta'].get('thunk_evals', 0)
        self.thunk_hits_total += example['meta'].get('thunk_hits', 0)
        self.recent_count += 1
        # Track pathological cases and runtime metrics
        if example['meta'].get('is_pathological', False):
            self.pathological_count += 1
        self.time_consumed_ratios.append(example['meta'].get('time_consumed_ratio', 0.0))
        self.size_growth_rates.append(example['meta'].get('size_growth_rate', 1.0))
        self.step_times_ms.append(example['meta'].get('step_ms', 0.0))
    
    def percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = int(len(sorted_vals) * p)
        return sorted_vals[min(k, len(sorted_vals) - 1)]
    
    def report(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = now - self.start_time
        recent_elapsed = now - self.last_time

        examples_per_sec = self.count / elapsed if elapsed > 0 else 0
        recent_rate = self.recent_count / recent_elapsed if recent_elapsed > 0 else 0
        tokens_per_sec = self.tokens / elapsed if elapsed > 0 else 0

        lat_list = list(self.latencies)

        total_thunk_ops = self.thunk_evals_total + self.thunk_hits_total
        share_rate = self.thunk_hits_total / total_thunk_ops if total_thunk_ops > 0 else 0

        # Compute pathological and runtime metrics
        pathological_rate = self.pathological_count / self.count if self.count > 0 else 0
        avg_time_ratio = sum(self.time_consumed_ratios) / len(self.time_consumed_ratios) if self.time_consumed_ratios else 0
        avg_growth = sum(self.size_growth_rates) / len(self.size_growth_rates) if self.size_growth_rates else 1.0
        avg_step_ms = sum(self.step_times_ms) / len(self.step_times_ms) if self.step_times_ms else 0

        return {
            'examples': self.count,
            'examples_per_sec': round(examples_per_sec, 1),
            'recent_rate': round(recent_rate, 1),
            'tokens_per_sec': round(tokens_per_sec, 1),
            'mean_size': round(sum(self.sizes) / len(self.sizes), 1) if self.sizes else 0,
            'mean_depth': round(sum(self.depths) / len(self.depths), 1) if self.depths else 0,
            'mean_steps': round(sum(self.steps) / len(self.steps), 1) if self.steps else 0,
            'diverged_rate': round(self.diverged / self.count, 3) if self.count > 0 else 0,
            'latency_p50': round(self.percentile(lat_list, 0.50) * 1000, 1),
            'latency_p95': round(self.percentile(lat_list, 0.95) * 1000, 1),
            'latency_p99': round(self.percentile(lat_list, 0.99) * 1000, 1),
            'thunk_evals': self.thunk_evals_total,
            'thunk_hits': self.thunk_hits_total,
            'share_hit_rate': round(share_rate, 3),
            # Runtime awareness metrics
            'pathological_count': self.pathological_count,
            'pathological_rate': round(pathological_rate, 3),
            'avg_time_consumed': round(avg_time_ratio, 3),
            'avg_step_ms': round(avg_step_ms, 2),
            'avg_size_growth': round(avg_growth, 2)
        }
    
    def reset_recent(self):
        self.last_time = time.time()
        self.recent_count = 0

# ============================================================================
# MODES
# ============================================================================

def live_mode(args, config: Config):
    #Streaming generation mode with rate limiting.#
    out_file = sys.stdout if args.out == '-' else open(args.out, 'w', buffering=1)
    metrics = Metrics()
    
    target_rate = args.rate
    min_interval = 1.0 / target_rate if target_rate > 0 else 0
    
    metrics_json_interval = None
    if hasattr(args, 'metrics_json') and args.metrics_json:
        try:
            metrics_json_interval = float(args.metrics_json.rstrip('s'))
        except ValueError:
            metrics_json_interval = 5.0
    
    sys.stderr.write(f"[Starting generation to {args.out}]\n")
    sys.stderr.write("[Generating first example...]\n")
    sys.stderr.flush()
    
    try:
        gen = yield_examples(config)
        last_metric_time = time.time()
        last_json_time = time.time()
        last_emit_time = time.time()
        last_progress_time = time.time()
        
        for example in gen:
            start_time = time.time()
            
            out_file.write(json.dumps(example) + '\n')
            out_file.flush()
            
            latency = time.time() - start_time
            metrics.update(example, latency)
            
            if metrics.count == 1:
                sys.stderr.write(f"[First example written successfully]\n")
                sys.stderr.flush()
            
            if time.time() - last_progress_time > 10.0 and metrics.count < 10:
                sys.stderr.write(f"\n[Progress: {metrics.count} examples generated so far]\n")
                sys.stderr.flush()
                last_progress_time = time.time()
            
            if time.time() - last_metric_time > 2.0:
                report = metrics.report()
                status = (f"[{report['examples']} ex | {report['recent_rate']:.1f}/s | "
                         f"size={report['mean_size']:.1f} depth={report['mean_depth']:.1f} "
                         f"steps={report['mean_steps']:.1f} | p50={report['latency_p50']:.1f}ms "
                         f"p95={report['latency_p95']:.1f}ms")

                if config.share:
                    status += f" | share={report['share_hit_rate']:.3f}"

                # Add pathological case warning if rate is high
                if report['pathological_rate'] > 0.05:  # >5% pathological
                    status += f" | ⚠️ path={report['pathological_rate']:.1%}"

                status += "]"
                sys.stderr.write(f"\r{status}")
                sys.stderr.flush()
                metrics.reset_recent()
                last_metric_time = time.time()
            
            if metrics_json_interval and time.time() - last_json_time > metrics_json_interval:
                report = metrics.report()
                report['timestamp'] = time.time()
                sys.stderr.write(f"\nMETRICS_JSON: {json.dumps(report)}\n")
                sys.stderr.flush()
                last_json_time = time.time()
            
            if target_rate > 0:
                elapsed = time.time() - last_emit_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_emit_time = time.time()
            
            if args.max_terms and metrics.count >= args.max_terms:
                break
    
    except KeyboardInterrupt:
        sys.stderr.write("\n[Interrupted]\n")
    finally:
        if out_file != sys.stdout:
            out_file.close()
        final_report = metrics.report()
        sys.stderr.write(f"\n[Complete: {final_report['examples']} examples | "
                        f"share_rate={final_report['share_hit_rate']:.3f} | "
                        f"pathological={final_report['pathological_count']} "
                        f"({final_report['pathological_rate']:.1%}) | "
                        f"avg_time={final_report['avg_time_consumed']:.1%} | "
                        f"avg_step={final_report['avg_step_ms']:.2f}ms | "
                        f"avg_growth={final_report['avg_size_growth']:.2f}x]\n")

def test_mode(args, config: Config):
    #Preview/test mode with span audit.#
    rng = random.Random(config.seed)
    
    if args.repl:
        print("REPL mode: commands :nf, :step, :span, :trace (parser not yet implemented)")
        return
    
    samples = []
    for i in range(args.n):
        example = generate_example(config, rng, i)
        if example:
            if isinstance(example, list):
                samples.extend(example[:1])
            else:
                samples.append(example)
    
    if not samples:
        print("No samples generated")
        return
    
    print(f"\n{'='*70}")
    print(f"FIRST SAMPLE SPAN AUDIT")
    print('='*70)
    first = samples[0]
    tokens = Renderer.tokenize(first['term'])
    print(f"Term: {first['term']}")
    print(f"Tokens: {' '.join(f'{i}:{tok}' for i, tok in enumerate(tokens))}")
    span = first['target_span']
    if span[0] < span[1]:
        highlighted = (first['term'][:span[0]] + 
                      '⟦' + first['term'][span[0]:span[1]] + '⟧' + 
                      first['term'][span[1]:])
        print(f"Redex: {highlighted}")
    print()
    
    if args.show_traces:
        print(f"{'='*70}")
        print("10-STEP TRACE")
        print('='*70)
        
        trace_config = Config(**{k: getattr(config, k) for k in 
                                ['strategy', 'render', 'max_depth', 'min_depth', 
                                 'max_size', 'max_steps', 'share', 'libraries', 
                                 'allow_divergent', 'seed']})
        trace_config.emit_trace = True
        
        rng_trace = random.Random(config.seed)
        trace_result = generate_example(trace_config, rng_trace, 0)
        
        if trace_result and isinstance(trace_result, list):
            for step_ex in trace_result[:10]:
                step = step_ex['step_k']
                term = step_ex['term']
                span = step_ex['target_span']
                
                if span[0] < span[1]:
                    highlighted = (term[:span[0]] + 
                                 '⟦' + term[span[0]:span[1]] + '⟧' + 
                                 term[span[1]:])
                else:
                    highlighted = term + " [NF]"
                
                print(f"Step {step:2d}: {highlighted[:70]}")
    
    if RICH_AVAILABLE and not args.no_ansi:
        console = Console()
        table = Table(title=f"Samples (n={len(samples)})", box=box.ROUNDED)
        table.add_column("Term")
        table.add_column("Steps", justify="right")
        table.add_column("Share", justify="right")
        
        for ex in samples[:15]:
            term_display = ex['term'][:50]
            share_info = ""
            if ex['meta'].get('thunk_hits', 0) + ex['meta'].get('thunk_evals', 0) > 0:
                total = ex['meta']['thunk_hits'] + ex['meta']['thunk_evals']
                rate = ex['meta']['thunk_hits'] / total
                share_info = f"{rate:.2f}"
            
            table.add_row(term_display, str(ex['steps_total']), share_info)
        console.print(table)

def validate_mode(args, config: Config):
    #Run comprehensive validation suite.#
    print(f"\n{'='*60}")
    print("LAMBDA CALCULUS VALIDATION SUITE")
    print('='*60)
    
    passed = 0
    failed = 0
    
    print("\n[1] Substitution soundness")
    rng = random.Random(42)
    subst_passed = 0
    for _ in range(20):
        gen = TermGenerator(rng, 5, 2, 20, [], False)
        body = gen.generate()
        arg = gen.generate()
        if body and arg:
            app_term = Term(TermType.APP, 
                          left=Term(TermType.ABS, body=body),
                          right=arg)
            
            reducer = TreeReducer(1)
            trace, _ = reducer.reduce(app_term)
            
            if len(trace) >= 2:
                result_term, _ = trace[1]
                ref_result = reference_substitute(body, 0, arg)
                
                if terms_alpha_equiv(result_term, ref_result):
                    subst_passed += 1
    
    if subst_passed >= 15:
        print(f"  ✓ PASS: {subst_passed}/20 substitutions verified")
        passed += 1
    else:
        print(f"  ✗ FAIL: only {subst_passed}/20 correct")
        failed += 1
    
    print("\n[2] Normal form idempotence")
    idem_passed = 0
    for _ in range(20):
        gen = TermGenerator(rng, 6, 2, 30, [], False)
        term = gen.generate()
        if term:
            reducer = TreeReducer(100)
            trace, diverged = reducer.reduce(term)
            
            if not diverged:
                nf, nf_redex = trace[-1]
                
                if nf_redex is None:
                    reducer2 = TreeReducer(1)
                    trace2, _ = reducer2.reduce(nf)
                    nf2, nf2_redex = trace2[-1]
                    
                    if nf2_redex is None and terms_alpha_equiv(nf, nf2):
                        idem_passed += 1
    
    if idem_passed >= 10:
        print(f"  ✓ PASS: {idem_passed} normal forms are idempotent")
        passed += 1
    else:
        print(f"  ✗ FAIL: only {idem_passed}/20 idempotent")
        failed += 1
    
    print("\n[3] Strategy equivalence")
    equiv_count = 0
    for _ in range(15):
        gen = TermGenerator(rng, 5, 2, 25, [], False)
        term = gen.generate()
        if term:
            tree_reducer = TreeReducer(50)
            tree_trace, tree_div = tree_reducer.reduce(term)
            
            graph_reducer = GraphReducer(50)
            graph_trace, graph_div, _, _ = graph_reducer.reduce(term)
            
            if not tree_div and not graph_div:
                tree_nf, _ = tree_trace[-1]
                graph_nf, _ = graph_trace[-1]
                
                if terms_alpha_equiv(tree_nf, graph_nf):
                    equiv_count += 1
    
    if equiv_count >= 10:
        print(f"  ✓ PASS: {equiv_count}/15 terms have equivalent NFs")
        passed += 1
    else:
        print(f"  ✗ FAIL: only {equiv_count}/15 equivalent")
        failed += 1
    
    print("\n[4] Span accuracy")
    span_ok = 0
    for _ in range(20):
        result = generate_example(config, rng, _)
        if result:
            # Handle both list and dict formats
            ex = result[0] if isinstance(result, list) else result
            span = ex['target_span']
            term_len = len(ex['term'])

            if ex['steps_total'] > 0:
                if 0 <= span[0] < span[1] <= term_len:
                    span_ok += 1
            else:
                if span[0] == 0 and span[1] == 0:
                    span_ok += 1
    
    if span_ok >= 15:
        print(f"  ✓ PASS: {span_ok}/20 spans valid")
        passed += 1
    else:
        print(f"  ✗ FAIL: only {span_ok}/20 valid spans")
        failed += 1
    
    print("\n[5] Normal-form final step span (regression guard)")
    # Critical: last step of terminating traces must have target_span == (0,0)
    # This catches the bug where redex_path or [] converts None to []
    nf_final_span_ok = 0
    for _ in range(20):
        result = generate_example(config, rng, _)
        if result and isinstance(result, list):
            # Check the LAST example in the trace
            last_ex = result[-1]
            # If this is the final step (step_k == steps_total) and not diverged
            if not last_ex['diverged'] and last_ex['step_k'] == last_ex['steps_total']:
                # Must have (0,0) span indicating normal form
                if last_ex['target_span'] == [0, 0]:
                    nf_final_span_ok += 1

    if nf_final_span_ok >= 15:
        print(f"  ✓ PASS: {nf_final_span_ok}/20 final NF steps have (0,0) span")
        passed += 1
    else:
        print(f"  ✗ FAIL: only {nf_final_span_ok}/20 have correct NF final span")
        failed += 1

    print("\n[6] Root-redex golden test (regression guard)")
    # Critical: (λx.x) 0 should reduce in exactly one step under both strategies
    # This catches the bug where if not redex_path: treats [] as falsy
    identity = Term(TermType.ABS, body=Term(TermType.VAR, var=0))
    arg = Term(TermType.VAR, var=0)
    root_term = Term(TermType.APP, left=identity, right=arg)

    # Test tree reducer
    tree_reducer = TreeReducer(10)
    tree_trace, tree_div = tree_reducer.reduce(root_term)
    tree_ok = (len(tree_trace) == 2 and not tree_div and
               tree_trace[-1][1] is None)  # Final step has no redex

    # Test graph reducer
    graph_reducer = GraphReducer(10)
    graph_trace, graph_div, _, _ = graph_reducer.reduce(root_term)
    graph_ok = (len(graph_trace) == 2 and not graph_div and
                graph_trace[-1][1] is None)  # Final step has no redex

    # Both should produce alpha-equivalent normal forms
    if tree_ok and graph_ok:
        tree_nf = tree_trace[-1][0]
        graph_nf = graph_trace[-1][0]
        equiv = terms_alpha_equiv(tree_nf, graph_nf)
        if equiv:
            print(f"  ✓ PASS: Root redex (λ.0)0 reduces correctly in both strategies")
            passed += 1
        else:
            print(f"  ✗ FAIL: Root redex normal forms differ between strategies")
            failed += 1
    else:
        status = f"tree={'OK' if tree_ok else 'FAIL'}, graph={'OK' if graph_ok else 'FAIL'}"
        print(f"  ✗ FAIL: Root redex did not reduce in one step ({status})")
        failed += 1

    print("\n[7] Renderer round-trip determinism")
    test_terms = []
    for _ in range(12):
        gen = TermGenerator(rng, 7, 3, 40, [], False)
        t = gen.generate()
        if t:
            test_terms.append(t)
    
    roundtrip_ok = 0
    for t in test_terms:
        r1 = Renderer.to_debruijn_with_spans(t)
        r2 = Renderer.to_debruijn_with_spans(t)
        if r1.string == r2.string and r1.spans == r2.spans:
            roundtrip_ok += 1
    
    if roundtrip_ok >= 10:
        print(f"  ✓ PASS: {roundtrip_ok}/12 renders are deterministic")
        passed += 1
    else:
        print(f"  ✗ FAIL: only {roundtrip_ok}/12 deterministic")
        failed += 1
    
    print("\n[8] Path encoding sanity")
    path_ok = 0
    for _ in range(20):
        gen = TermGenerator(rng, 6, 2, 30, [], False)
        term = gen.generate()
        if term:
            reducer = TreeReducer(5)
            trace, _ = reducer.reduce(term)

            for t, path in trace:
                # path is None means NF, empty list means redex at root
                if path is not None:
                    node_id = path_to_node_id(path)
                    r_db = Renderer.to_debruijn_with_spans(t)
                    r_named = Renderer.to_named_with_spans(t)

                    if node_id in r_db.spans and node_id in r_named.spans:
                        path_ok += 1
                        break
    
    if path_ok >= 15:
        print(f"  ✓ PASS: {path_ok}/20 paths encode correctly")
        passed += 1
    else:
        print(f"  ✗ FAIL: only {path_ok}/20 correct")
        failed += 1
    
    print(f"\n{'='*60}")
    if failed == 0:
        print(f"ALL TESTS PASSED ({passed}/8 test groups)")
        print('='*60)
        return True
    else:
        print(f"FAILURES: {failed} test groups failed")
        print('='*60)
        return False

# ============================================================================
# MULTIPROCESSING
# ============================================================================

def worker_process(worker_id: int, config: Config, queue: mp.Queue, count_target: Optional[int]):
    #Worker process for parallel example generation.#
    # Each worker gets its own RNG seeded differently
    base_seed = config.seed if config.seed else 42
    worker_seed = base_seed + worker_id * 10000
    rng = random.Random(worker_seed)

    draw_index = worker_id
    examples_generated = 0

    try:
        while True:
            if count_target and examples_generated >= count_target:
                break

            example = generate_example(config, rng, draw_index)
            if example:
                if isinstance(example, list):
                    # Full trace - send all examples
                    for ex in example:
                        queue.put(('example', ex))
                        examples_generated += 1
                else:
                    queue.put(('example', example))
                    examples_generated += 1

            draw_index += mp.cpu_count()  # Interleave draw indices across workers

    except KeyboardInterrupt:
        pass
    finally:
        queue.put(('done', worker_id))


def live_mode_mp(args, config: Config, num_workers: int = None):
    #Multiprocessing version of live mode for high throughput.#
    if num_workers is None:
        num_workers = min(8, mp.cpu_count())  # Default to 8 or CPU count

    out_file = sys.stdout if args.out == '-' else open(args.out, 'w', buffering=1)
    metrics = Metrics()

    count_per_worker = None
    if args.max_terms:
        count_per_worker = (args.max_terms + num_workers - 1) // num_workers

    sys.stderr.write(f"[Starting {num_workers}-process generation to {args.out}]\n")
    sys.stderr.flush()

    # Create queue and workers
    queue = mp.Queue(maxsize=num_workers * 100)
    workers = []

    for i in range(num_workers):
        p = mp.Process(target=worker_process, args=(i, config, queue, count_per_worker))
        p.start()
        workers.append(p)

    active_workers = num_workers
    last_metric_time = time.time()

    try:
        while active_workers > 0:
            msg_type, data = queue.get()

            if msg_type == 'example':
                t0 = time.time()
                out_file.write(json.dumps(data) + '\n')
                metrics.update(data, time.time() - t0)

                # Periodic metrics
                if time.time() - last_metric_time > 5.0:
                    report = metrics.report()
                    sys.stderr.write(
                        f"\r[{report['examples']} ex | "
                        f"{report['examples_per_sec']:.1f}/s | "
                        f"steps:{report['mean_steps']:.1f} | "
                        f"size:{report['mean_size']:.1f}]"
                    )
                    sys.stderr.flush()
                    last_metric_time = time.time()
                    metrics.reset_recent()

                if args.max_terms and metrics.count >= args.max_terms:
                    break

            elif msg_type == 'done':
                active_workers -= 1

    except KeyboardInterrupt:
        sys.stderr.write("\n[Interrupted]\n")
    finally:
        for p in workers:
            p.terminate()
            p.join(timeout=1.0)

        if out_file != sys.stdout:
            out_file.close()

        final_report = metrics.report()
        sys.stderr.write(f"\n\n[Final: {final_report['examples']} examples "
                        f"at {final_report['examples_per_sec']:.1f}/s]\n")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='λ-calculus training data generator')
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    live = subparsers.add_parser('live')
    live.add_argument('--strategy', default='normal', choices=['normal', 'levy_like'])
    live.add_argument('--render', default='debruijn', choices=['debruijn', 'named'])
    live.add_argument('--rate', type=int, default=50, help='Target examples/sec (0=unlimited)')
    live.add_argument('--max-depth', type=int, default=8)
    live.add_argument('--min-depth', type=int, default=2)
    live.add_argument('--max-size', type=int, default=50)
    live.add_argument('--wall-clock-limit-ms', type=float, default=100.0,
                     help='Wall clock time limit per term in milliseconds (primary limiter)')
    live.add_argument('--max-steps', type=int, default=10000,
                     help='Maximum reduction steps (safety fallback)')
    live.add_argument('--max-terms', type=int)
    live.add_argument('--out', default='train.jsonl')
    live.add_argument('--seed', type=int)
    live.add_argument('--share', action='store_true')
    live.add_argument('--allow-divergent', action='store_true')
    live.add_argument('--emit-next', action='store_true')
    live.add_argument('--emit-nf', action='store_true')
    live.add_argument('--emit-trace', action='store_true')
    live.add_argument('--include', default='')
    live.add_argument('--buffer-size', type=int, default=1, help='Deprecated: flush is now automatic')
    live.add_argument('--no-ansi', action='store_true')
    live.add_argument('--metrics-json', type=str, help='Emit JSON metrics every N seconds (e.g., 5s)')
    live.add_argument('--workers', type=int, help='Number of parallel workers (enables multiprocessing mode)')
    
    test = subparsers.add_parser('test')
    test.add_argument('--n', type=int, default=20)
    test.add_argument('--strategy', default='normal', choices=['normal', 'levy_like'])
    test.add_argument('--render', default='debruijn', choices=['debruijn', 'named'])
    test.add_argument('--max-depth', type=int, default=8)
    test.add_argument('--min-depth', type=int, default=2)
    test.add_argument('--wall-clock-limit-ms', type=float, default=100.0,
                     help='Wall clock time limit per term in milliseconds')
    test.add_argument('--max-steps', type=int, default=10000,
                     help='Maximum reduction steps (safety fallback)')
    test.add_argument('--seed', type=int, default=1337)
    test.add_argument('--share', action='store_true')
    test.add_argument('--show-traces', action='store_true')
    test.add_argument('--repl', action='store_true')
    test.add_argument('--no-ansi', action='store_true')
    
    validate = subparsers.add_parser('validate')
    validate.add_argument('--n', type=int, default=2000)
    validate.add_argument('--max-depth', type=int, default=10)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = Config()
    if hasattr(args, 'strategy'):
        config.strategy = args.strategy
    if hasattr(args, 'render'):
        config.render = args.render
    if hasattr(args, 'max_depth'):
        config.max_depth = args.max_depth
    if hasattr(args, 'min_depth'):
        config.min_depth = args.min_depth
    if hasattr(args, 'max_size'):
        config.max_size = args.max_size
    if hasattr(args, 'wall_clock_limit_ms'):
        config.wall_clock_limit_ms = args.wall_clock_limit_ms
    if hasattr(args, 'max_steps'):
        config.max_steps = args.max_steps
    if hasattr(args, 'seed'):
        config.seed = args.seed
    if hasattr(args, 'share'):
        config.share = args.share
    if hasattr(args, 'allow_divergent'):
        config.allow_divergent = args.allow_divergent
    if hasattr(args, 'emit_next'):
        config.emit_next = args.emit_next
    if hasattr(args, 'emit_nf'):
        config.emit_nf = args.emit_nf
    if hasattr(args, 'emit_trace'):
        config.emit_trace = args.emit_trace
    if hasattr(args, 'include') and args.include:
        libs = [lib.strip() for lib in args.include.split(',')]
        if 'y' in libs and not config.allow_divergent:
            sys.stderr.write("[Warning: Y combinator requires --allow-divergent, excluding]\n")
            libs = [l for l in libs if l != 'y']
        config.libraries = libs
    
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.write("λ-Calculus Training Data Generator v2.0\n")
    sys.stderr.write(f"Strategy: {config.strategy}")
    if config.share:
        sys.stderr.write(" [SHARING ENABLED]")
    sys.stderr.write(f" | Depth: {config.max_depth}\n")
    
    if args.mode == 'live':
        sys.stderr.write(f"Output: {args.out} | Rate: {args.rate}/s")
        if args.max_terms:
            sys.stderr.write(f" | Target: {args.max_terms} examples")
        sys.stderr.write("\n")
        
        if config.min_depth >= config.max_depth:
            sys.stderr.write("[WARNING: min_depth >= max_depth, generation may fail]\n")
        if config.max_size < 10:
            sys.stderr.write("[WARNING: max_size is very small, generation may be slow]\n")
        if not config.allow_divergent and config.max_steps < 20:
            sys.stderr.write("[WARNING: max_steps is low and divergent terms blocked]\n")
    
    sys.stderr.write("=" * 60 + "\n")
    sys.stderr.flush()

    if args.mode == 'live':
        if hasattr(args, 'workers') and args.workers:
            live_mode_mp(args, config, num_workers=args.workers)
        else:
            live_mode(args, config)
    elif args.mode == 'test':
        test_mode(args, config)
    elif args.mode == 'validate':
        success = validate_mode(args, config)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()