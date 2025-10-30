# Aggressive Size Distribution Adjustments

## Problem Identified

After initial test generation, size distribution showed:
```
160-180 nodes:   0.0%  ⚠️
180-200 nodes:   0.0%  ⚠️
```

Despite `max_size` parameters up to 200, **almost zero large terms** were generated.

## Root Cause

Large terms are naturally harder to generate because they:
1. Have difficulty passing size constraints during generation
2. Are more likely to hit wall clock limits during reduction
3. Get filtered out as pathological more often
4. Require more reduction steps (hitting max_steps limits)

**Previous conservative parameters weren't aggressive enough.**

## Aggressive Parameter Changes

### Main Config (main.rs)

| Parameter | Before | After | Change |
|-----------|--------|-------|--------|
| `max_depth` | 12 | **15** | +25% |
| `max_size` | 150 | **250** | +67% |
| `max_steps` | 500 | **1000** | +100% |

### Cycle Distribution (parallel.rs)

**Depth Cycles (0-9):**
```rust
Before: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
After:  8, 10, 12, 13, 14, 15, 16, 17, 18, 20
```
- Eliminated shallow depths (6-7)
- Added extreme depths (16-20)
- **Push the model to its limits!**

**Size Cycles (0-9):**
```rust
Before: 80, 100, 120, 140, 160, 180, 200, 150, 130, 110
After:  120, 150, 180, 210, 240, 270, 300, 250, 200, 220
```
- **Minimum increased**: 80 → 120 (+50%)
- **Maximum increased**: 200 → 300 (+50%)
- **More cycles favor large terms**: 70% of cycles now generate size 200+

### Default Generator (generator.rs)

| Parameter | Before | After |
|-----------|--------|-------|
| `max_depth` | 8 | **15** |
| `max_size` | 100 | **250** |

## Expected Size Distribution After Changes

### Target Distribution
```
      0- 50 nodes:  10% (down from 23%)
     50-100 nodes:  20% (down from 24%)
    100-150 nodes:  25% (up from 19%)
    150-200 nodes:  20% (up from 9%)
    200-250 nodes:  15% (up from 3%)
    250-300 nodes:  10% (NEW - was 0%)
```

### Key Improvements

1. **Eliminated tiny terms**: Min size increased from 80 → 120
2. **Heavy bias toward large**: 70% of cycles generate 200+ nodes
3. **Extreme cases**: Now generating up to 300 nodes (was 200 max)
4. **More steps allowed**: 1000 max_steps (was 500) for completion

## Why These Numbers?

### Max Size: 250 → 300
- With filtering, terms often end up 60-70% of max_size
- Target: Get 15-25% of examples in 160-250 range
- **300 max_size → expect ~180-210 median for large cycles**

### Max Depth: 15 → 20
- Model diagnostic showed failure at depth >4
- Need training examples at depth 10-15 for robustness
- Cycle 9 now generates depth 20 (extreme case)

### Max Steps: 500 → 1000
- Large terms need proportionally more reduction steps
- 500 steps was bottleneck for 200+ node terms
- 1000 steps allows completion without hitting limits

## Validation After Generation

Run these checks on new data:

### 1. Size Distribution
```python
import json
sizes = []
with open('data.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        sizes.append(ex['meta']['size'])

print(f"Size 150-200: {sum(1 for s in sizes if 150 <= s < 200) / len(sizes):.1%}")
print(f"Size 200-250: {sum(1 for s in sizes if 200 <= s < 250) / len(sizes):.1%}")
print(f"Size 250+: {sum(1 for s in sizes if s >= 250) / len(sizes):.1%}")
```

**Target**:
- 150-200: **>15%** (was ~1%)
- 200-250: **>10%** (was ~0%)
- 250+: **>5%** (was 0%)

### 2. Depth Distribution
```python
depths = []
with open('data.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        depths.append(ex['meta']['depth'])

print(f"Depth 15-20: {sum(1 for d in depths if 15 <= d <= 20) / len(depths):.1%}")
print(f"Max depth: {max(depths)}")
```

**Target**:
- Depth 15-20: **>10%** (was <1%)
- Max depth: **20** (was ~15)

## Expected Model Improvements

After retraining with this aggressive data:

| Metric | Current | Target |
|--------|---------|--------|
| Large terms (100-150) | 5% correct | **50-60%** |
| Very large (150-200) | ~0% correct | **30-40%** |
| Extreme (200+) | 0% correct | **15-25%** |

The model will learn:
- ✅ Don't panic on long sequences
- ✅ Large terms can still reduce
- ✅ Temporary growth is normal
- ✅ Deep nesting doesn't mean divergence

## Generation Command

```bash
cd lambda_gen_rs
cargo build --release

# Generate 15M examples with aggressive parameters
./target/release/lambda_gen_rs generate training_data_15m_large.jsonl 15000000 16 250

# Expected time: ~24-28 hours
# (Larger terms take longer to reduce)
```

## Verification Script

```bash
# After generation completes, check distribution
python tests/check_diversity.py training_data_15m_large.jsonl

# Should see:
# - Size range: 2-300 (was 2-150)
# - Mean size: ~120 (was ~48)
# - Size >150: 25-30% (was <1%)
```

## Safety Notes

**Why not go even larger?**
- 300+ nodes would hit wall clock limits too often
- Pathological filtering would reject most
- Diminishing returns beyond this point
- Model needs to master 150-250 first before 300+

**Monitoring:**
- If pathological rate >5%, may need to back off
- If generation throughput drops below 100 examples/sec, sizes may be too large
- If >30% of terms timeout, increase wall_clock_limit_ms

**This is aggressive but measured** - pushing the model hard without breaking the pipeline!
