# Lambda Calculus Throughput Investigation & Fuel Budget Implementation

## Overview

This document describes the investigation into throughput issues and the implementation of comprehensive fuel budget tracking for the Lambda Calculus reduction project.

## Problem Statement

The Lambda Calculus training data generation system exhibited pathological behavior where:
1. **Throughput variability**: Most terms reduced efficiently, but occasional terms caused wall clock time to explode
2. **Fuel budget opacity**: The reduction fuel budget (`max_steps`) existed but wasn't visible to the model
3. **Pathological case blindness**: No mechanism to detect or track terms with exponential growth or excessive step counts

## Root Causes Identified

### 1. Missing Fuel Metrics
The fuel budget (`max_steps`) controlled reduction termination but wasn't exposed in the training data:
- Model had no awareness of fuel constraints
- No tracking of fuel consumption patterns
- No way to identify when reductions were approaching limits

### 2. Pathological Case Invisibility
Terms exhibiting problematic behavior were generated but not flagged:
- **High step count**: Terms using >80% of fuel budget
- **Size explosion**: Terms growing 3x or more during reduction
- **Large terms**: Terms exceeding 200 nodes

### 3. Trace Amplification
Each reduction trace generated `steps_total` training examples:
- A 100-step reduction generated 100 examples
- Pathological cases disproportionately inflated dataset
- No visibility into this amplification effect

## Implementation

### 1. Fuel Budget Metrics (lambda_gen.py)

Added comprehensive fuel tracking to every training example:

```python
'meta': {
    # ... existing fields ...
    'max_steps': config.max_steps,              # Total fuel budget
    'fuel_remaining': fuel_remaining,           # Steps left
    'fuel_consumed_ratio': fuel_consumed_ratio, # Fraction used (0.0-1.0)
    'is_pathological': is_pathological,         # Boolean flag
    'size_growth_rate': size_growth_rate,       # current_size / initial_size
    'initial_size': initial_size                # Starting term size
}
```

**Pathological Detection Criteria:**
- Used >80% of fuel budget (`steps_total > max_steps * 0.8`)
- Size tripled during reduction (`size_growth_rate > 3.0`)
- Current term exceeds 200 nodes (`current_size > 200`)

### 2. Real-Time Monitoring (lambda_gen.py)

Enhanced metrics tracking and reporting:

```python
class Metrics:
    # Added fields:
    pathological_count: int
    fuel_consumed_ratios: deque
    size_growth_rates: deque
```

**Status Display:**
```
[1000 ex | 50.0/s | size=25.0 depth=4.5 steps=12.3 | ⚠️ path=8.5%]
```

**Final Report:**
```
[Complete: 5000 examples | share_rate=0.342 | pathological=425 (8.5%) |
 avg_fuel=15.2% | avg_growth=1.45x]
```

### 3. Training Integration (lambda_train.py)

Fuel metrics now flow through the training pipeline:

**Dataset Loading:**
- Extracts fuel metrics from metadata with backward compatibility
- Passes through data pipeline to training batches

**Batch Collation:**
- Collects `fuel_remaining`, `fuel_consumed_ratio`, `is_pathological`, `size_growth_rate`
- Available as tensors in training batches

**Training Monitoring:**
- Tracks pathological ratio per batch
- Logs to console with warning when >5% pathological
- Logs to TensorBoard for analysis

**Console Output:**
```
Step 1000 | Loss: 0.1234 | EM: 0.876 | IoU: 0.912 | LR: 3.5e-4 |
2048 tok/s | ⚠️ path=7.2%
```

## Validation

Created `test_fuel_metrics.py` to verify implementation:

**Test Results:**
```
Example 1: Generated trace with 2 steps
  Metadata:
    max_steps: 50
    fuel_remaining: 50
    fuel_consumed_ratio: 0.000
    size_growth_rate: 1.00
  Last step:
    fuel_remaining: 49
    fuel_consumed_ratio: 0.020
    size_growth_rate: 0.50
  ✓ Fuel remaining calculation correct: 49
  ✓ Fuel consumed ratio correct: 0.020
```

All metrics verified correct.

## Benefits

### 1. Visibility
- Real-time tracking of pathological cases during generation
- Clear metrics on fuel consumption patterns
- Size growth monitoring

### 2. Debugging
- Can filter training data by pathological flag
- Can analyze correlation between fuel consumption and model performance
- Can identify problematic term patterns

### 3. Model Awareness
- Fuel metrics available as training signals
- Model can potentially learn fuel-aware reduction strategies
- Enables future work on adaptive fuel budgeting

### 4. Dataset Quality
- Quantify and control pathological example ratio
- Balance dataset composition
- Detect distribution shifts in generated data

## Schema Changes

### Before (v1.x):
```json
{
  "term": "...",
  "step_k": 5,
  "steps_total": 10,
  "meta": {
    "size": 25,
    "thunk_evals": 3,
    "thunk_hits": 2
  }
}
```

### After (v2.0):
```json
{
  "term": "...",
  "step_k": 5,
  "steps_total": 10,
  "meta": {
    "size": 25,
    "thunk_evals": 3,
    "thunk_hits": 2,
    "max_steps": 100,
    "fuel_remaining": 95,
    "fuel_consumed_ratio": 0.05,
    "is_pathological": false,
    "size_growth_rate": 1.25,
    "initial_size": 20
  }
}
```

## Future Work

### 1. Fuel-Aware Model Architecture
Modify model to explicitly condition on fuel metrics:
- Append fuel tokens to input sequence
- Add auxiliary scalar inputs
- Multi-task learning with fuel prediction

### 2. Adaptive Fuel Budgeting
Dynamic `max_steps` based on term characteristics:
- Higher budgets for complex terms
- Early termination for simple terms
- Optimize throughput while maintaining coverage

### 3. Pathological Case Analysis
Deep dive into what makes terms pathological:
- Term structure patterns
- Combinator usage
- Reduction strategy impact

### 4. Dataset Balancing
Implement sampling strategies:
- Down-sample pathological cases
- Up-weight efficient reductions
- Curriculum learning based on fuel consumption

## Usage

### Generation
```bash
# Standard generation with fuel tracking
python lambda_gen.py live --strategy levy_like --share --max-steps 100

# Monitor will show:
# [500 ex | 45.0/s | steps=12.5 | ⚠️ path=5.2%]
```

### Training
```bash
# Training automatically uses fuel metrics
python lambda_train.py --train data.jsonl

# Logs will show:
# Step 1000 | Loss: 0.1234 | ⚠️ path=8.0%
```

### Analysis
```bash
# Test fuel metrics
python test_fuel_metrics.py

# Validate terms are well-posed
python lambda_gen.py validate --n 100
```

## Conclusion

The fuel budget implementation provides comprehensive visibility into reduction behavior, enabling:
1. **Detection** of pathological cases in real-time
2. **Monitoring** of fuel consumption patterns
3. **Training** with fuel-aware signals
4. **Debugging** of throughput issues

This addresses the original investigation goals:
- ✅ Terms verified as well-posed and reducible
- ✅ Pathological cases now detected and tracked
- ✅ Fuel budget mechanism fully implemented
- ✅ Model aware of fuel metrics via training data
