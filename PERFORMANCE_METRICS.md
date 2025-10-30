# Performance Metrics Documentation

## Overview

The performance metrics system provides comprehensive analysis of the lambda calculus model's reduction capabilities, comparing it against the gold-standard Levy graph reduction.

## Quick Start

```bash
# Run with defaults (200 terms)
./run_performance_test.sh

# Run with custom number of terms
./run_performance_test.sh 500

# Run with custom output directory
./run_performance_test.sh 1000 results/my_analysis
```

Or run directly:
```bash
python performance_metrics.py \
    --checkpoint runs/levy700m/checkpoints/step_10000.pt \
    --num-terms 200 \
    --output-dir results/performance_analysis
```

## Metrics Collected

### 1. Term Characteristics

For each test term, we analyze:
- **Size**: Total number of nodes (variables, abstractions, applications)
- **Depth**: Maximum nesting level
- **Complexity**: Size × Depth
- **Redex count**: Number of reducible expressions
- **Initial redex location**: Path to first redex in normal order

### 2. Reduction Metrics (Per Step)

For each reduction step:
- **Term size/depth evolution**: How the term changes
- **Redex selection**: Path chosen for reduction
- **Redex depth**: How nested the selected redex is
- **Inference time**: Time taken to predict this step (model only)
- **NF confidence**: Model's confidence that term is in normal form

### 3. Comparison Metrics (Model vs Gold)

For each term:
- **Correctness**: Does model reach the same normal form?
- **Steps**: Number of reductions to reach normal form
- **Tokens**: Total characters processed
- **Time**: Inference time (model only)
- **Divergence**: When do strategies differ?
- **Divergence type**: Why do they differ?
  - `path_choice`: Different redex selected
  - `early_stop`: Model stopped before reaching normal form
  - `invalid_pred`: Model made invalid prediction

### 4. Aggregated Statistics

#### Overall Performance
- **Convergence rate**: % of terms reaching normal form
- **Correctness rate**: % reaching correct normal form
- **Exact match rate**: % following exact same reduction path as gold
- **Model faster rate**: % where model uses fewer steps

#### Performance by Term Type
Breaking down performance by:
- **Size**: small (<20), medium (20-50), large (50-80), xlarge (>80)
- **Depth**: shallow (<4), medium (4-7), deep (>7)
- **Complexity**: simple, moderate, complex, very_complex

For each category:
- Correct rate
- Faster rate
- Average step difference

#### Path Pattern Analysis
- **Common patterns**: Most frequent path prefixes (e.g., "LLR", "RRL")
- **Left/right bias**: Tendency to reduce left vs right branches
- **Root reductions**: Frequency of reducing at root
- **Deep reductions**: Frequency of reducing nested redexes
- **Unique model patterns**: Patterns model uses that gold doesn't

#### Speed Analysis
- **Fastest terms**: Terms where model is significantly faster
- **Slowest terms**: Terms where model is slower
- **Speedup characteristics**: What makes a term fast/slow for the model

#### Error Analysis
- **Error cases**: Terms where model fails
- **Failure patterns**: Common characteristics of failures
- **Divergence points**: Where model deviates from gold

## Output Files

### 1. `summary_metrics.json`

High-level statistics:
```json
{
  "overall": {
    "total_terms": 200,
    "correctness_rate": 0.985,
    "model_faster_rate": 0.642,
    "avg_step_difference": -1.23,
    "avg_time_per_inference_ms": 12.5
  },
  "performance_by_size": {
    "small": {"correct_rate": 0.99, "faster_rate": 0.75},
    "medium": {"correct_rate": 0.98, "faster_rate": 0.60}
  },
  "path_patterns": {
    "model": {"root": 45, "LLL": 23, "LLR": 18},
    "gold": {"root": 52, "LLL": 28, "LRR": 15}
  },
  "speed_analysis": {
    "fastest_terms": [...],
    "slowest_terms": [...]
  }
}
```

### 2. `detailed_comparisons.json`

Per-term analysis:
```json
[
  {
    "term_id": 0,
    "initial_term": "\\.(\\.(0))(\\.(1))",
    "characteristics": {
      "size": 15,
      "depth": 4,
      "num_redexes": 2,
      "complexity_score": 60.0
    },
    "model": {
      "steps": 8,
      "tokens": 234,
      "time_ms": 95.3,
      "avg_time_ms": 11.9,
      "size_evolution": [15, 13, 11, 9, 7, 5, 3, 1],
      "depth_evolution": [4, 3, 3, 2, 2, 1, 1, 1]
    },
    "gold": {
      "steps": 10,
      "tokens": 287,
      "size_evolution": [15, 14, 12, 10, 8, 6, 4, 2, 1, 1]
    },
    "comparison": {
      "model_correct": true,
      "model_faster_steps": true,
      "step_difference": -2,
      "diverged": true,
      "divergence_step": 3,
      "divergence_type": "path_choice"
    }
  }
]
```

### 3. `sample_traces.json`

Step-by-step reduction traces for interesting cases:
```json
[
  {
    "term_id": 0,
    "initial_term": "\\.(\\.(0))(\\.(1))",
    "model_steps": [
      {
        "step": 0,
        "term": "\\.(\\.(0))(\\.(1))",
        "size": 15,
        "depth": 4,
        "redex_path": [0],
        "redex_depth": 1,
        "inference_time_ms": 12.3,
        "nf_confidence": 0.02
      },
      {
        "step": 1,
        "term": "\\.(0)(\\.(1))",
        "size": 13,
        "depth": 3,
        "redex_path": [],
        "redex_depth": 0,
        "inference_time_ms": 11.8,
        "nf_confidence": 0.05
      }
    ],
    "gold_steps": [...]
  }
]
```

## Interpreting Results

### What Makes the Model Fast?

Look at `fastest_term_types` in summary:
- Do fast terms share common characteristics (size, depth, complexity)?
- Check `path_patterns` to see if model uses different strategies
- Review `sample_traces.json` to see step-by-step what model does differently

### What Types of Terms Can It Reduce Well?

Check `performance_by_size`, `performance_by_depth`, `performance_by_complexity`:
- High `correct_rate` = model handles these well
- High `faster_rate` = model is efficient on these
- Negative `avg_step_diff` = model uses fewer steps

### Are There Unexpected Superior Strategies?

Look for:
- `unique_model_patterns`: Paths model uses that gold doesn't
- Terms with high speedup in `fastest_terms`
- Divergence type `path_choice` where model still reaches correct NF

Check corresponding traces in `sample_traces.json` to see what the model is doing differently.

### Where Does the Model Struggle?

Check `error_analysis`:
- `error_cases`: Specific failures
- `common_failure_patterns`: What types of terms fail
- `divergence_type_distribution`: Why model diverges

### How Fast Is Inference?

Check `avg_time_per_inference_ms` in overall stats:
- This is time per reduction step
- Compare to total reduction time vs gold's step count
- Consider: is faster inference worth potentially more steps?

## Example Analysis Workflow

1. **Run the test**:
   ```bash
   ./run_performance_test.sh 500 results/baseline
   ```

2. **Check overall correctness**:
   - Look at `correctness_rate` in console output
   - Should be >95% for good model

3. **Analyze speed**:
   - Check `model_faster_rate` and `avg_step_difference`
   - Review `fastest_term_types` to understand what model excels at

4. **Investigate strategies**:
   - Compare `model_path_patterns` vs `gold_path_patterns`
   - Check `unique_model_patterns` for novel approaches
   - Open `sample_traces.json` to see specific examples

5. **Debug errors**:
   - If correctness < 95%, check `error_cases`
   - Look at `common_failure_patterns`
   - Review error traces to understand failure modes

## Advanced Usage

### Custom Term Generation

Modify the parameters:
```bash
python performance_metrics.py \
    --checkpoint runs/model/checkpoint.pt \
    --num-terms 1000 \
    --min-depth 3 \
    --max-depth 15 \
    --max-size 200 \
    --output-dir results/large_terms
```

### Verbose Output

See term-by-term analysis in real-time:
```bash
python performance_metrics.py \
    --checkpoint runs/model/checkpoint.pt \
    --num-terms 50 \
    --verbose \
    --output-dir results/debug
```

### CPU-Only Mode

If CUDA unavailable:
```bash
python performance_metrics.py \
    --checkpoint runs/model/checkpoint.pt \
    --device cpu \
    --num-terms 100 \
    --output-dir results/cpu_test
```

## Metrics Reference

| Metric | Range | Interpretation |
|--------|-------|----------------|
| `correctness_rate` | 0-1 | % reaching correct normal form (higher = better) |
| `exact_match_rate` | 0-1 | % matching gold path exactly (lower ok if correct) |
| `model_faster_rate` | 0-1 | % using fewer steps than gold (higher = more efficient) |
| `avg_step_difference` | -∞ to +∞ | Average step difference (negative = faster) |
| `divergence_rate` | 0-1 | % diverging from gold (lower if correctness high) |
| `avg_time_per_inference_ms` | 0-∞ | Inference speed (lower = faster) |
| `speedup` | -∞ to +∞ | (gold_steps - model_steps) / gold_steps |

## Tips

1. **Run multiple times**: RNG affects term generation, run 3-5 times and average
2. **Use enough terms**: 200+ for reliable statistics, 1000+ for publishable results
3. **Watch for outliers**: Check `sample_traces` for extreme speedup/slowdown cases
4. **Compare checkpoints**: Run on multiple checkpoints to track training progress
5. **Correlate with training**: Compare metrics to training accuracy/loss curves
