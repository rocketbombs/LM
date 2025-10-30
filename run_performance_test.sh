#!/bin/bash
#
# Run comprehensive performance metrics test
#
# This script automatically finds the latest checkpoint and runs
# the performance metrics analysis on it.
#
# Usage:
#   ./run_performance_test.sh [num_terms] [output_dir]
#
# Examples:
#   ./run_performance_test.sh                    # Default: 200 terms
#   ./run_performance_test.sh 500                # 500 terms
#   ./run_performance_test.sh 1000 results/full  # 1000 terms, custom output dir

set -e

# Configuration
NUM_TERMS=${1:-200}
OUTPUT_DIR=${2:-results/performance_$(date +%Y%m%d_%H%M%S)}
MAX_DEPTH=10
MAX_SIZE=100
MAX_STEPS=1000

# Find latest checkpoint
echo "Finding latest checkpoint..."
CHECKPOINT=$(find runs -name "*.pt" -type f | grep -v "optimizer" | sort -V | tail -n 1)

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in runs/ directory"
    echo "Please train a model first using lambda_train.py"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo "Number of terms: $NUM_TERMS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run performance analysis
python performance_metrics.py \
    --checkpoint "$CHECKPOINT" \
    --num-terms "$NUM_TERMS" \
    --max-depth "$MAX_DEPTH" \
    --max-size "$MAX_SIZE" \
    --max-steps "$MAX_STEPS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Performance analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/summary_metrics.json      (Overall statistics)"
echo "  - $OUTPUT_DIR/detailed_comparisons.json (Per-term analysis)"
echo "  - $OUTPUT_DIR/sample_traces.json        (Step-by-step traces)"
