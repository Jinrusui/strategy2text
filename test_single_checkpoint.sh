#!/bin/bash

# Test script to run HVA-X analysis on a single checkpoint
# Usage: ./test_single_checkpoint.sh [checkpoint_name]

set -e

# Configuration
CHECKPOINT_DIR="checkpoint_videos"
RESULTS_BASE_DIR="test_checkpoint_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to first checkpoint if none specified
CHECKPOINT_NAME="${1:-steps_00199992}"
CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_NAME"
OUTPUT_DIR="$RESULTS_BASE_DIR/$CHECKPOINT_NAME"

echo "🧪 Testing HVA-X Analysis on Single Checkpoint"
echo "Checkpoint: $CHECKPOINT_NAME"
echo "Video directory: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

# Count videos
video_count=$(find "$CHECKPOINT_PATH" -name "*.mp4" | wc -l)
echo "📹 Found $video_count video files"

if [ "$video_count" -eq 0 ]; then
    echo "⚠️  No video files found, exiting..."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🔄 Running Phase 2A: Event Detection (Direct Mode)"
echo "----------------------------------------"

# Phase 2A
python "$SCRIPT_DIR/run_phase2a_only.py" \
    --video-dir "$CHECKPOINT_PATH" \
    --direct-mode \
    --output-dir "$OUTPUT_DIR" \
    --output-prefix "phase2a_${CHECKPOINT_NAME}" \
    --verbose

# Find Phase 2A output
phase2a_file=$(find "$OUTPUT_DIR" -name "phase2a_${CHECKPOINT_NAME}_*.json" | head -1)
if [ -z "$phase2a_file" ]; then
    echo "❌ Phase 2A output file not found"
    exit 1
fi

echo ""
echo "🔄 Running Phase 2B: Guided Analysis"
echo "----------------------------------------"

# Phase 2B
python "$SCRIPT_DIR/run_phase2b_only.py" \
    --phase2a-file "$phase2a_file" \
    --output-dir "$OUTPUT_DIR" \
    --output-prefix "phase2b_${CHECKPOINT_NAME}" \
    --verbose

# Find Phase 2B output
phase2b_file=$(find "$OUTPUT_DIR" -name "phase2b_${CHECKPOINT_NAME}_*.json" | head -1)
if [ -z "$phase2b_file" ]; then
    echo "❌ Phase 2B output file not found"
    exit 1
fi

echo ""
echo "🔄 Running Phase 3: Meta-Synthesis with Video References"
echo "----------------------------------------"

# Phase 3
python "$SCRIPT_DIR/run_phase3_only.py" \
    --phase2b-file "$phase2b_file" \
    --output-dir "$OUTPUT_DIR" \
    --output-prefix "phase3_${CHECKPOINT_NAME}" \
    --save-report \
    --verbose

echo ""
echo "✅ Test completed successfully!"
echo "📁 Results saved to: $OUTPUT_DIR"

# Show generated files
echo ""
echo "📄 Generated files:"
find "$OUTPUT_DIR" -type f | sort | while read -r file; do
    rel_path=$(realpath --relative-to="." "$file")
    echo "  - $rel_path"
done

# Show report preview if available
report_file=$(find "$OUTPUT_DIR" -name "*_report_*.md" | head -1)
if [ -n "$report_file" ]; then
    echo ""
    echo "📖 Report preview (first 10 lines):"
    echo "----------------------------------------"
    head -10 "$report_file"
    echo "----------------------------------------"
    echo "📄 Full report: $report_file"
fi 