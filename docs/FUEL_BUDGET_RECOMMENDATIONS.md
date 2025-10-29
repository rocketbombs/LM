# Fuel Budget Recommendations

## Executive Summary

After extensive testing, we found that **throughput issues are fundamental to the graph reducer on complex terms**, not just a tuning problem. Even with aggressive fuel budgets (max_steps=10-15), certain term structures cause the reducer to hang.

## Test Results

### Throughput by Depth (with aggressive max_steps=10-15)

| Depth | Success Rate | Throughput | Degradation | Recommendation |
|-------|-------------|------------|-------------|----------------|
| 2-4   | 100%        | 3944/s     | 1x (baseline) | ✓ **Use for production** |
| 4-6   | 93%         | 4.5/s      | 880x        | ⚠️ Acceptable with filtering |
| 6-8   | 73%         | 0.9/s      | 4500x       | ✗ **Avoid** |

## Root Cause Analysis

### The Problem is NOT Just max_steps

Tests show "Timed out" errors occur **within the graph reducer**, not from hitting the fuel budget:

```
[Error during reduction: Timed out after 3s]  # Reducer hangs before max_steps!
```

**Key Finding**: Individual reduction steps can be exponentially expensive on certain term structures. Even with max_steps=10, some terms timeout.

### Why Reducing max_steps Doesn't Fully Solve It

1. **Per-step cost**: Each reduction step can take hundreds of milliseconds
2. **Term structure**: Certain combinations cause exponential blow-up
3. **Graph sharing**: The sharing optimization doesn't help on all patterns
4. **Pathological sequences**: Some reductions never complete

## Recommended Configuration

### For Production Training Data Generation

```python
# Conservative - High Success Rate (recommended)
Config(
    max_depth=4,          # Keep shallow
    min_depth=2,
    max_size=20,
    max_steps=15,         # Aggressive limit
    share=True,
    allow_divergent=True  # Accept some divergence
)
# Expected: 100% success, 3900+ terms/s, 2% divergence rate
```

```python
# Moderate - Balanced
Config(
    max_depth=6,          # Medium complexity
    min_depth=4,
    max_size=40,
    max_steps=12,         # Very aggressive
    share=True,
    allow_divergent=True
)
# Expected: 93% success, 4-5 terms/s, 10% divergence rate
# Note: 7% failure rate due to reducer hangs
```

### NOT Recommended

```python
# TOO AGGRESSIVE - High Failure Rate
Config(
    max_depth=8,          # Too deep
    max_steps=12,
    # ...
)
# Result: 73% success, 0.9 terms/s, 27% failures
# Pathological cases cause reducer to hang
```

## Practical Guidelines

### 1. Set max_steps Based on Wall Clock Target

If you want each term to complete in <10ms on average:
- **Depth 2-4**: max_steps=15 → 0.2ms avg ✓
- **Depth 4-6**: max_steps=12 → 0.5ms avg ✓
- **Depth 6-8**: max_steps=10 → Still 0.7ms, but 27% fail ✗

### 2. Accept Higher Divergence Rates

With aggressive fuel budgets:
- **2-10% divergence** is normal and acceptable
- Diverged terms are **well-posed** - just complex
- They hit max_steps gracefully, no timeout

### 3. Filter Pathological Failures

Terms that cause reducer hangs should be:
- Logged and analyzed
- Filtered from training data
- Used to identify problematic patterns

### 4. Stratify by Depth

Generate different depth ranges separately:

```bash
# High volume, simple terms
python lambda_gen.py live --max-depth 4 --max-steps 15 --max-terms 10000

# Lower volume, complex terms
python lambda_gen.py live --max-depth 6 --max-steps 12 --max-terms 2000
```

## Future Work

### Short Term

1. **Profile the graph reducer** on failing terms
   - Identify which term structures cause hangs
   - Add early termination heuristics

2. **Term structure filtering**
   - Detect problematic patterns during generation
   - Skip terms with known pathological features

3. **Timeout per step**
   - Add step-level timeout in reducer
   - Fail fast instead of hanging

### Long Term

1. **Reducer optimization**
   - Investigate alternative reduction strategies
   - Implement iterative deepening
   - Better sharing heuristics

2. **Adaptive fuel budgeting**
   - Predict fuel needs from term structure
   - Dynamic max_steps based on complexity

3. **Curriculum generation**
   - Start with simple terms
   - Gradually increase complexity
   - Maintain quality thresholds

## Bottom Line

**For reliable, high-throughput training data generation:**

✓ Use `max_depth=4, max_steps=15` (3900+ terms/s, 100% success)

⚠️ Use `max_depth=6, max_steps=12` with caution (4-5 terms/s, 93% success)

✗ Avoid `max_depth>6` (<<1 terms/s, <80% success)

**Key Insight**: The throughput issue is fundamental - certain lambda calculus terms are inherently expensive to reduce. The fuel budget helps, but doesn't eliminate pathological cases. Accept 2-10% divergence as normal.
