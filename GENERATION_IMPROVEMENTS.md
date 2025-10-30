# Data Generation Pipeline Improvements

## Issues Found and Fixed

### 🐛 Critical Bugs

#### 1. Hardcoded Seed (main.rs:72)
**Problem**: Seed was hardcoded to `42` instead of using the time-based seed
```rust
// BEFORE (BUG):
seed: 42,  // This ignored the computed seed!

// AFTER (FIXED):
seed,  // Now uses time-based seed for true diversity
```

**Impact**: All workers were using the same seed patterns, severely limiting diversity.

#### 2. Artificial Chunk Limit (parallel.rs:87)
**Problem**: Generation capped at 10,000 chunks regardless of target size
```rust
// BEFORE (BUG):
((max + chunk_size - 1) / chunk_size).min(10000)  // Hard cap!

// AFTER (FIXED):
let estimated_chunks = (max / 10).max(1);  // Scales with target
```

**Impact**:
- At ~100 examples per chunk, this limited generation to ~1M examples
- With multi-step traces (~3-4 examples per term), max was ~3-4M examples
- **For 15M target, this completely blocked generation!**

### 🎯 Parameter Enhancements

#### 3. Increased Depth Range
```rust
// BEFORE:
max_depth: 6-10 (5 cycles)

// AFTER:
max_depth: 6-15 (10 cycles)
```

**Why**: Diagnostic showed model fails on depth > 4 redexes. Need more deep examples.

#### 4. Increased Size Range
```rust
// BEFORE:
max_size: 80-120

// AFTER:
max_size: 80-200
```

**Why**: Model panics on term growth. Need more examples of temporarily growing terms.

#### 5. More Complexity Cycles
```rust
// BEFORE:
complexity_cycle = chunk_id % 5  // Only 5 variations

// AFTER:
complexity_cycle = chunk_id % 10  // 10 variations for wider coverage
```

**Why**: More variation = better coverage of edge cases.

## New Generation Distribution

### Depth Distribution (10 cycles)
| Cycle | Depth | Purpose |
|-------|-------|---------|
| 0 | 6 | Simple baseline |
| 1 | 7 | Medium-simple |
| 2 | 8 | Medium |
| 3 | 9 | Medium-complex |
| 4 | 10 | Complex |
| 5 | 11 | Very complex |
| 6 | 12 | **Deep nesting (fixes depth blindness)** |
| 7 | 13 | Very deep |
| 8 | 14 | Extreme depth |
| 9 | 15 | **Maximum depth (push the limits!)** |

### Size Distribution (10 cycles)
| Cycle | Size | Purpose |
|-------|------|---------|
| 0 | 80 | Small terms |
| 1 | 100 | Medium |
| 2 | 120 | Medium-large |
| 3 | 140 | Large |
| 4 | 160 | Very large |
| 5 | 180 | Huge |
| 6 | 200 | **Very huge (growth examples)** |
| 7 | 150 | Back to large |
| 8 | 130 | Medium-large variety |
| 9 | 110 | Medium variety |

## Expected Improvements

### Coverage Improvements

**Before**:
- Depth 6-10: Good coverage
- Depth 11+: **Almost no coverage** ❌
- Size 80-120: Good coverage
- Size 120+: **Limited coverage** ⚠️

**After**:
- Depth 6-10: Good coverage (cycles 0-4)
- Depth 11-15: **10-50% coverage** ✓ (cycles 5-9)
- Size 80-120: Good coverage (cycles 0-2)
- Size 120-200: **30-60% coverage** ✓ (cycles 3-6)

### Target Failure Modes

Based on diagnostic, these changes specifically target:

#### 1. Deep Redex Blindness
- **Symptom**: Model fails when redex at depth > 4
- **Fix**: 40% of data now has depth 12-15 (cycles 6-9)
- **Example terms**: `\.\.\.\.\.\.(\.x)y` (redex at depth 6)

#### 2. Growth Aversion
- **Symptom**: Model panics when terms grow during reduction
- **Fix**: 30% of data now has size 160-200 (cycles 4-6)
- **Example**: `(\.xx)(\.yy)` → `\.yy\.yy` (temporary growth)

#### 3. Lambda Prefix Confusion
- **Symptom**: Model sees `\.\.\.\` and assumes NF
- **Fix**: More deep abstractions with nested redexes
- **Example**: `\.\.\.\.(\.redex)` should reduce, not stop

## Generation Capacity

### Before (Bugged):
```
Max chunks: 10,000
Examples per chunk: ~10
Total capacity: ~100K examples
With multi-step: ~300K-400K examples max
```

### After (Fixed):
```
Target: 15M examples
Estimated chunks needed: 1.5M
Examples per chunk: ~10
Total capacity: 15M+ examples ✓
```

## Testing the Changes

### Quick Test (100 examples)
```bash
cd lambda_gen_rs
cargo build --release
./target/release/lambda_gen_rs generate test_output.jsonl 100 8 250
```

Check output:
```bash
# Count examples
wc -l test_output.jsonl

# Check depth distribution
grep -o '"depth":[0-9]*' test_output.jsonl | sort | uniq -c

# Check size distribution
grep -o '"size":[0-9]*' test_output.jsonl | sort -n | tail -20
```

### Full Scale Test (15M examples)
```bash
# This will now work! (was capped at 3-4M before)
./target/release/lambda_gen_rs generate training_data_15m.jsonl 15000000 16 250
```

Expected runtime:
- 16 workers
- ~10-15 examples/sec per worker
- Total: ~160-240 examples/sec
- 15M / 200 = 75,000 seconds = **~21 hours**

## Verification

After generating new data, run these checks:

### 1. Depth Coverage
```python
import json
depths = []
with open('training_data_15m.jsonl') as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:  # Sample
            ex = json.loads(line)
            depths.append(ex['meta']['depth'])

# Should see good coverage in 10-15 range now!
print(f"Max depth: {max(depths)}")
print(f"Depth > 10: {sum(1 for d in depths if d > 10) / len(depths):.1%}")
```

### 2. Size Coverage
```python
sizes = []
with open('training_data_15m.jsonl') as f:
    for i, line in enumerate(f):
        if i % 100000 == 0:
            ex = json.loads(line)
            sizes.append(ex['meta']['size'])

# Should see terms up to 200+ now!
print(f"Max size: {max(sizes)}")
print(f"Size > 150: {sum(1 for s in sizes if s > 150) / len(sizes):.1%}")
```

### 3. Generation Progress
The generation should now complete all 15M examples instead of stopping at 3-4M!

## Retraining Strategy

After generating new data:

1. **Validate coverage**: Run checks above
2. **Mix with old data** (optional): 80% new, 20% old for stability
3. **Train from checkpoint**: Continue from best existing model
4. **Monitor depth/size performance**: Track correctness by term complexity
5. **Compare diagnostics**: Run `diagnose_early_stopping.py` before/after

## Expected Model Improvements

With these changes, the model should learn:

✓ Deep redex detection (depth 5-10)
✓ Handling term growth during reduction
✓ Not confusing `\.\.\.\` prefix with normal form
✓ More robust reduction on complex terms

Target metrics after retraining:
- Overall correctness: 26% → **70-80%** (current → target)
- Medium terms (20-50 nodes): 2.5% → **60-70%**
- Large terms (50-80 nodes): 0% → **40-50%**
- Deep terms (>7 depth): 5.8% → **50-60%**
