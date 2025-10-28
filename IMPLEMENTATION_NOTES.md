# Implementation Progress - Graph Reducer Improvements

## Completed ✅

### 1. Critical Bug Fixes (Committed: b941101)

- **Root-redex misclassification fix**
  - Changed `if not redex_path:` → `if redex_path is None:`
  - Correctly distinguishes `None` (normal form) from `[]` (root redex)
  - Verified with golden test: `(λ.0)0` now reduces correctly

- **NF span labeling fix**
  - Changed `redex_path or []` → `redex_path` 
  - Prevents converting `None` to `[]`, preserving correct `(0,0)` span
  - Final steps of terminating traces now correctly labeled

### 2. Regression Test Infrastructure

- **NF final span test**: Verifies last step has `(0,0)` span
- **Root-redex golden test**: Verifies `(λ.0)0` reduces in one step
- **Test count**: Expanded from 6 to 8 test groups

## Design (Not Yet Implemented)

### 1. Structural Sharing Metrics

**Approach**: Charge sharing at β-reduction time
- Count occurrences of binder (de Bruijn index 0) in body
- Track: `bind_edges`, `bind_edges_saved`, `shared_bindings`

**Implementation Started**:
```python
def _count_binder_uses(self, term: Term, target_index: int = 0, depth: int = 0) -> int:
    """Count occurrences of a de Bruijn index in a term."""
    # Counts syntactic duplication pressure that sharing collapses
```

**Issue Encountered**:
- Projecting graph to tree (`graph.to_tree()`) at each β-reduction is expensive
- Can cause performance issues for complex graphs with many reduction steps
- **Solution Needed**: Direct graph traversal instead of tree projection

**Test Case Works**:
```python
# (λ. 0 0) a - duplicating body
# Expected: bind_edges=2, bind_edges_saved=1, shared_bindings=1
# Actual: ✓ All metrics correct
```

### 2. Fuel-Based Evaluation (Not Implemented)

**Design**:
- Replace wall-time heuristics with step-indexed evaluator
- Fuel counter: consume one unit per β-reduction
- Return status: `{normal_form, timeout, safety_cap}`
- Budget formula: `c * size(term) * (1 + depth(term))`

**Schema Changes Needed**:
```python
{
  'budget': int,
  'status': str,  # HALT_NF, HALT_TIMEOUT, HALT_SAFETY
  'fuel_left': int,
  'fuel_used': int,
  'end_action': str,
  'next_redex_span': Optional[List[int]]  # For timeout cases
}
```

**Benefits**:
- Machine-independent semantics (model can learn)
- Total, deterministic evaluation
- Natural training patterns (step-wise + one-shot)

### 3. Data Gates (Not Implemented)

**Invariants to Enforce**:
1. Final step must have `(0,0)` span
2. Terminating graph runs must agree with tree normal forms  
3. Spans must be in bounds
4. For duplicating bodies: `bind_edges ≥ 2`, `bind_edges_saved ≥ 1`

## Next Steps

1. **Fix Structural Sharing Performance**
   - Implement direct graph traversal for binder counting
   - Avoid expensive `to_tree()` projections
   - Add cycle detection

2. **Implement Fuel-Based Evaluation**
   - Add fuel parameter to both reducers
   - Track fuel_left, fuel_used
   - Return comprehensive status

3. **Add Data Gates**
   - Validation filters in generation pipeline
   - Ensure all invariants hold
   - Log violations for debugging

4. **Polish Items**
   - Memoized thunks for faithful projection
   - Actually populate family size metrics
   - Separate gen_latency from io_latency

## Key Insight

The reviewer's design is excellent and mathematically rigorous:
- **Structural sharing**: Measured where it happens (at β-reduction)
- **Fuel-based eval**: Total, deterministic, learnable
- **Data gates**: Ensure supervision signal integrity

Implementation requires care around performance - tree projections are expensive for DAGs.
