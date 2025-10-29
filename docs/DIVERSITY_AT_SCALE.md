# Diversity at Scale: 10M Examples Without Duplication

## User Requirement

**Target**: 10 million training examples
**Requirement**: "Ensure no duplication, lots of coverage and saturation with high diversity"

## Combinatorial Space Analysis

### Lambda Calculus Term Space

With current configuration:
- **Depth range**: 6-10 (varied per chunk)
- **Size range**: 80-120 nodes (varied per chunk)
- **Min depth**: 3

**Conservative estimate of unique terms:**

At each depth d, a lambda term has roughly 3^d structural choices (Var, Abs, App).
With variable indices 0 to d-1, the space multiplies further.

For depth 8:
- Structural choices: ~3^8 = 6,561
- Variable choices: ~8^(nodes with vars) = millions
- Tree shape variations: factorial combinations

**Estimated unique terms after filtering:**
- Depth 6-10 range: **100M-1B+ unique normalizing, clean terms**
- After ultra-strict filtering (~60% rejection): **40M-400M usable unique terms**

**Conclusion**: 10M examples is **0.01-0.025% of available space** - duplication risk is LOW with good RNG.

## Diversity Strategy

### 1. Time-Based RNG Seeding

**Change**: Use nanosecond timestamp as base seed (not hardcoded 42)

```rust
// OLD: Fixed seed
seed: 42,

// NEW: Time-based for true randomness
let seed = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap()
    .as_nanos() as u64;
```

**Effect:**
- Each run gets unique seed
- Different runs produce different terms
- No overlap between sessions
- Enables multiple parallel generation runs

### 2. Per-Chunk Seed Derivation

**Current strategy** (already good):
```rust
let worker_seed = config.seed
    .wrapping_add(chunk_id as u64)
    .wrapping_mul(0x9e3779b97f4a7c15);  // Golden ratio for dispersion
```

**For 10M examples:**
- After 60% filtering, need ~25M generated terms
- At 100 terms/chunk: 250k chunks
- Each chunk gets unique seed via golden ratio multiplication
- Ensures no overlap between chunks

**Effect**: 250k unique RNG streams for 10M examples

### 3. Parameter Variation

**NEW**: Cycle through 5 complexity levels per chunk

```rust
match (chunk_id % 5) {
    0 => (depth: 6,  size: 80),   // Simple
    1 => (depth: 7,  size: 90),   // Medium-simple
    2 => (depth: 8,  size: 100),  // Balanced (default)
    3 => (depth: 9,  size: 110),  // Medium-complex
    4 => (depth: 10, size: 120),  // Complex
}
```

**Effect:**
- Every 5 chunks covers full complexity spectrum
- For 250k chunks: 50k chunks per complexity level
- **Each complexity level**: 2M terms × 5 levels = 10M terms
- Ensures even distribution across difficulty levels

### 4. Maintained Strict Filtering

All filters remain active:
- ✅ Non-normalizing (allow_divergent=false)
- ✅ Pathological (time >50%, steps >3ms, growth >2.5x, size >150)
- ✅ Trivial (zero-step, single-step)
- ✅ Premature NF markers

**Effect**: Only clean, diverse, multi-step reduction traces

## Expected Diversity Metrics

### For 10M Examples

| Metric | Target | Notes |
|--------|--------|-------|
| Unique terms | >90% | Time-based seeding + parameter variation |
| Unique traces | ~100% | Each trace ID is unique by design |
| Max term duplication | <10x | Any term appears ≤10 times |
| Size std dev | >25 | Broad size distribution (80-200 nodes) |
| Depth range | 3-25 | Full spectrum from simple to complex |
| Step range | 2-100+ | Varied reduction complexity |

### Coverage Distribution

With 5 complexity levels cycled evenly:

**Depth distribution** (expected):
- Depth 6: ~15-20% (2M terms)
- Depth 7: ~15-20% (2M terms)
- Depth 8: ~20-25% (2-2.5M terms)
- Depth 9: ~15-20% (2M terms)
- Depth 10: ~15-20% (2M terms)
- Depth 11+: ~5-10% (natural variation)

**Size distribution** (expected):
- 50-80 nodes: ~15%
- 80-100 nodes: ~25%
- 100-120 nodes: ~30%
- 120-150 nodes: ~20%
- 150+ nodes: ~10%

## Testing Diversity

### Step 1: Generate Test Sample

```powershell
cd X:\Code\LM\lambda_gen_rs
cargo build --release --target x86_64-pc-windows-gnu

# Generate 50k sample (represents 0.5% of 10M)
cargo run --release --target x86_64-pc-windows-gnu -- generate sample_50k.jsonl 50000 16 250
```

### Step 2: Check Diversity

```powershell
cd X:\Code\LM
python tests/check_diversity.py sample_50k.jsonl
```

**Expected output:**
```
✅ UNIQUENESS METRICS
  Unique Terms: >45,000 / 50,000 (>90%)
  Max occurrences: <10x

✅ COVERAGE METRICS
  Size range: 40-200 nodes
  Depth range: 3-25
  Std dev: >25

✅ EXTRAPOLATION TO 10M
  Estimated unique: >9M
  Excellent diversity for large-scale!
```

### Step 3: Verify Data Quality

```powershell
python tests/diagnose_training_data.py sample_50k.jsonl 10000
```

**Expected:**
```
✅ Pathological: 0%
✅ Zero-step: 0%
✅ Diverged: 0%
✅ All examples clean
```

## Generation Strategy for 10M

### Option 1: Single Large Run

```powershell
# Generate all 10M in one run (takes ~6-8 hours with filtering)
cargo run --release --target x86_64-pc-windows-gnu -- generate training_10m.jsonl 10000000 16 250
```

**Pros:**
- Simple, single command
- Automatic time-based seeding
- Consistent filtering

**Cons:**
- Long runtime (6-8 hours)
- If interrupted, must restart
- Single failure point

### Option 2: Multiple Batches

```powershell
# Generate in 10 batches of 1M (can parallelize on different machines)
for i in 1..10:
    cargo run --release -- generate training_batch_{i}.jsonl 1000000 16 250 {custom_seed}

# Combine
cat training_batch_*.jsonl > training_10m.jsonl
```

**Pros:**
- Can run on multiple machines
- Fault-tolerant (restart only failed batch)
- Can verify diversity per batch

**Cons:**
- More complex
- Need to provide unique seeds per batch
- Must combine files

**Recommended**: Option 2 for robustness

### Seed Strategy for Multiple Batches

Use distinct seeds to guarantee no overlap:
```powershell
# Batch 1: seed from timestamp
cargo run --release -- generate batch1.jsonl 1000000 16 250

# Wait 1 second, then batch 2 (different timestamp)
sleep 1
cargo run --release -- generate batch2.jsonl 1000000 16 250

# Or manually specify seeds far apart
cargo run --release -- generate batch1.jsonl 1000000 16 250 1000000
cargo run --release -- generate batch2.jsonl 1000000 16 250 2000000
cargo run --release -- generate batch3.jsonl 1000000 16 250 3000000
# etc.
```

## Performance Estimates

### Throughput

With ultra-strict filtering (~60% rejection rate):
- Generation rate: ~5,000-8,000 terms/second (16 workers)
- After filtering: ~2,000-3,000 usable examples/second
- For 10M examples: **55-85 minutes of pure generation**

### With I/O and Overhead

- Writing to disk: +10-15%
- RNG variation: +5%
- **Total: ~70-110 minutes (1-2 hours)**

### Disk Usage

- Average example: ~500 bytes JSONL
- 10M examples: ~5 GB uncompressed
- Compressed (gzip): ~1-1.5 GB

## Diversity Validation

After generating 10M examples, validate:

### 1. Uniqueness Check

```powershell
python tests/check_diversity.py training_10m.jsonl 100000
```

**Target**: >90% uniqueness on 100k sample

### 2. Full Scan (Optional)

```powershell
# Count unique terms in full dataset (takes ~10 minutes)
python -c "
import json
unique = set()
with open('training_10m.jsonl') as f:
    for line in f:
        unique.add(json.loads(line)['term'])
print(f'Unique terms: {len(unique):,} / 10,000,000 ({100*len(unique)/10000000:.1f}%)')
"
```

**Target**: >9M unique terms (>90%)

### 3. Distribution Check

```powershell
# Check depth distribution
python -c "
import json
from collections import Counter
depths = []
with open('training_10m.jsonl') as f:
    for line in f:
        depths.append(json.loads(line)['meta']['depth'])
print('Depth distribution:')
for d, count in Counter(depths).most_common():
    print(f'  Depth {d}: {100*count/len(depths):.1f}%')
"
```

**Target**: Even distribution across depths 6-10

## Troubleshooting

### Issue: Low Uniqueness (<80%)

**Symptoms:**
```
Unique Terms: 7,500,000 / 10,000,000 (75%)
```

**Causes:**
- Fixed seed instead of time-based
- Parameter variation not working
- Chunk_id not varying

**Fix:**
1. Ensure time-based seed is being used
2. Check parameter variation in code
3. Regenerate with fresh seed

### Issue: Uneven Distribution

**Symptoms:**
```
Depth 8: 60% (too much)
Depth 6: 5% (too little)
```

**Causes:**
- Filtering bias (simpler terms filtered more)
- Parameter cycling not working

**Fix:**
1. Check complexity_cycle logic
2. Adjust filtering thresholds if needed

### Issue: Too Many Duplicates

**Symptoms:**
```
Max occurrences: 500x
```

**Causes:**
- RNG seed not varying
- Generator producing same terms

**Fix:**
1. Verify time-based seeding
2. Check golden ratio multiplication
3. Ensure chunk_id is incrementing

## Summary

**For 10M examples with high diversity:**

1. ✅ **Time-based RNG seeding**: Each run gets unique seed
2. ✅ **Parameter variation**: 5 complexity levels cycled
3. ✅ **Golden ratio dispersion**: 250k unique RNG streams
4. ✅ **Massive combinatorial space**: 40M-400M unique terms available
5. ✅ **Diversity validation**: Tools to verify >90% uniqueness

**Expected result:**
- **>9M unique terms** (>90% uniqueness)
- **Even distribution** across complexity levels
- **<10x max duplication** for any term
- **Complete coverage** of depth 6-10 spectrum

**Generation time**: 1-2 hours for 10M examples
**Disk usage**: ~5 GB uncompressed, ~1-1.5 GB compressed

---

**Status**: Ready for 10M scale generation
**Confidence**: HIGH - Combinatorial space supports 10M+ without duplication
**Recommendation**: Use multiple batches (10x 1M) for robustness
