#!/bin/bash
# Verification script for example-level pathological filtering
# This script generates a small dataset and verifies 0% pathological

set -e

echo "========================================"
echo "Example-Level Filtering Verification"
echo "========================================"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LAMBDA_GEN_DIR="$PROJECT_DIR/lambda_gen_rs"
TEST_FILE="$PROJECT_DIR/test_zero_pathological.jsonl"
NUM_EXAMPLES=5000
NUM_WORKERS=8
WALL_CLOCK=250

echo "Configuration:"
echo "  Test file: $TEST_FILE"
echo "  Examples: $NUM_EXAMPLES"
echo "  Workers: $NUM_WORKERS"
echo "  Wall clock: ${WALL_CLOCK}ms"
echo ""

# Clean up old test file
if [ -f "$TEST_FILE" ]; then
    echo "Removing old test file..."
    rm "$TEST_FILE"
fi

# Build the generator
echo "Building Rust generator..."
cd "$LAMBDA_GEN_DIR"
cargo build --release --quiet
echo "✅ Build complete"
echo ""

# Generate test data
echo "Generating $NUM_EXAMPLES examples..."
echo "This may take 30-60 seconds..."
cargo run --release --quiet -- generate "$TEST_FILE" "$NUM_EXAMPLES" "$NUM_WORKERS" "$WALL_CLOCK"
echo "✅ Generation complete"
echo ""

# Count examples
ACTUAL_COUNT=$(wc -l < "$TEST_FILE")
echo "Generated: $ACTUAL_COUNT examples"
echo ""

# Run diagnostic
echo "Running diagnostic analysis..."
echo "========================================"
cd "$PROJECT_DIR"
python tests/diagnose_training_data.py "$TEST_FILE" 10000

echo ""
echo "========================================"
echo "Verification Complete"
echo "========================================"
echo ""
echo "Expected Results:"
echo "  ✅ Pathological: 0 / $ACTUAL_COUNT (0.0%)"
echo "  ✅ Diverged: 0 / $ACTUAL_COUNT (0.0%)"
echo "  ✅ Premature NF: 0 / $ACTUAL_COUNT (0.0%)"
echo "  ✅ Zero-step: 0 / $ACTUAL_COUNT (0.0%)"
echo "  ✅ Mean size: 60-90 nodes"
echo "  ✅ Max growth: <2.5x"
echo ""
echo "If any pathological cases appear, the fix is not working correctly."
echo "All metrics should show 0% for filtering to be successful."
