#!/bin/bash

# HVA-X Analysis Script for All Checkpoint Folders
# Runs Phase 2A (direct mode), 2B, and 3 for each checkpoint in /checkpoint_videos
# Saves all results with video-referenced reports

# Configuration
CHECKPOINT_DIR="checkpoint_videos"
RESULTS_BASE_DIR="checkpoint_analysis_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to extract checkpoint name from path
get_checkpoint_name() {
    basename "$1"
}

# Function to run HVA-X analysis for a single checkpoint
run_checkpoint_analysis() {
    local checkpoint_path="$1"
    local checkpoint_name=$(get_checkpoint_name "$checkpoint_path")
    local output_dir="$RESULTS_BASE_DIR/$checkpoint_name"
    local checkpoint_start=$(date +%s)  # Add this line to track start time
    
    print_status "Processing checkpoint: $checkpoint_name"
    print_status "Video directory: $checkpoint_path"
    print_status "Output directory: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Check if videos exist in the checkpoint directory
    if [ ! -d "$checkpoint_path" ]; then
        print_error "Checkpoint directory not found: $checkpoint_path"
        return 1
    fi
    
    # Count video files
    video_count=$(find "$checkpoint_path" -name "*.mp4" | wc -l)
    if [ "$video_count" -eq 0 ]; then
        print_warning "No video files found in $checkpoint_path, skipping..."
        return 0
    fi
    
    print_status "Found $video_count video files in $checkpoint_name"
    
    # Phase 2A: Event Detection (Direct Mode)
    print_status "Running Phase 2A: Event Detection (Direct Mode)"
    phase2a_start=$(date +%s)
    
    if python "$SCRIPT_DIR/run_phase2a_only.py" \
        --video-dir "$checkpoint_path" \
        --direct-mode \
        --output-dir "$output_dir" \
        --output-prefix "phase2a_${checkpoint_name}" \
        --verbose; then
        
        phase2a_end=$(date +%s)
        phase2a_duration=$((phase2a_end - phase2a_start))
        print_success "Phase 2A completed in ${phase2a_duration}s"
        
        # Find the generated Phase 2A file
        phase2a_file=$(find "$output_dir" -name "phase2a_${checkpoint_name}_*.json" | head -1)
        if [ -z "$phase2a_file" ]; then
            print_error "Phase 2A output file not found"
            return 1
        fi
        print_status "Phase 2A results: $phase2a_file"
        
    else
        print_error "Phase 2A failed for $checkpoint_name"
        return 1
    fi
    
    # Phase 2B: Guided Analysis
    print_status "Running Phase 2B: Guided Analysis"
    phase2b_start=$(date +%s)
    
    if python "$SCRIPT_DIR/run_phase2b_only.py" \
        --phase2a-file "$phase2a_file" \
        --output-dir "$output_dir" \
        --output-prefix "phase2b_${checkpoint_name}" \
        --verbose; then
        
        phase2b_end=$(date +%s)
        phase2b_duration=$((phase2b_end - phase2b_start))
        print_success "Phase 2B completed in ${phase2b_duration}s"
        
        # Find the generated Phase 2B file
        phase2b_file=$(find "$output_dir" -name "phase2b_${checkpoint_name}_*.json" | head -1)
        if [ -z "$phase2b_file" ]; then
            print_error "Phase 2B output file not found"
            return 1
        fi
        print_status "Phase 2B results: $phase2b_file"
        
    else
        print_error "Phase 2B failed for $checkpoint_name"
        return 1
    fi
    
    # Phase 3: Meta-Synthesis
    print_status "Running Phase 3: Meta-Synthesis with Video References"
    phase3_start=$(date +%s)
    
    if python "$SCRIPT_DIR/run_phase3_only.py" \
        --phase2b-file "$phase2b_file" \
        --output-dir "$output_dir" \
        --output-prefix "phase3_${checkpoint_name}" \
        --save-report \
        --verbose; then
        
        phase3_end=$(date +%s)
        phase3_duration=$((phase3_end - phase3_start))
        print_success "Phase 3 completed in ${phase3_duration}s"
        
        # Find the generated report
        report_file=$(find "$output_dir" -name "phase3_${checkpoint_name}_report_*.md" | head -1)
        if [ -n "$report_file" ]; then
            print_success "Final report generated: $report_file"
        fi
        
    else
        print_error "Phase 3 failed for $checkpoint_name"
        return 1
    fi
    
    # Calculate total time
    total_end=$(date +%s)
    total_duration=$((total_end - checkpoint_start))
    
    print_success "✅ Checkpoint $checkpoint_name completed successfully!"
    print_success "   Total time: ${total_duration}s (2A: ${phase2a_duration}s, 2B: ${phase2b_duration}s, 3: ${phase3_duration}s)"
    print_success "   Results saved to: $output_dir"
    
    echo "----------------------------------------"
    return 0
}

# Main execution
main() {
    print_status "Starting HVA-X Analysis for All Checkpoints"
    print_status "Checkpoint directory: $CHECKPOINT_DIR"
    print_status "Results will be saved to: $RESULTS_BASE_DIR"
    echo "========================================"
    
    # Check if checkpoint directory exists
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        print_error "Checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi
    
    # Create base results directory
    mkdir -p "$RESULTS_BASE_DIR"
    
    # Find all checkpoint subdirectories
    checkpoints=($(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "steps_*" | sort))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        print_error "No checkpoint directories found in $CHECKPOINT_DIR"
        exit 1
    fi
    
    print_status "Found ${#checkpoints[@]} checkpoint directories:"
    for checkpoint in "${checkpoints[@]}"; do
        print_status "  - $(get_checkpoint_name "$checkpoint")"
    done
    echo "========================================"
    
    # Process each checkpoint
    success_count=0
    failure_count=0
    overall_start=$(date +%s)
    
    for checkpoint in "${checkpoints[@]}"; do
        if run_checkpoint_analysis "$checkpoint"; then
            ((success_count++))
        else
            ((failure_count++))
            print_error "Failed to process $(get_checkpoint_name "$checkpoint")"
        fi
    done
    
    # Final summary
    overall_end=$(date +%s)
    overall_duration=$((overall_end - overall_start))
    
    echo "========================================"
    print_status "🎯 HVA-X ANALYSIS COMPLETE"
    print_status "Total checkpoints processed: ${#checkpoints[@]}"
    print_success "Successful: $success_count"
    if [ $failure_count -gt 0 ]; then
        print_error "Failed: $failure_count"
    fi
    print_status "Total processing time: ${overall_duration}s"
    print_status "Results directory: $RESULTS_BASE_DIR"
    
    # List generated reports
    echo ""
    print_status "Generated Reports:"
    find "$RESULTS_BASE_DIR" -name "*_report_*.md" | while read -r report; do
        rel_path=$(realpath --relative-to="." "$report")
        print_success "  📄 $rel_path"
    done
    
    if [ $failure_count -eq 0 ]; then
        print_success "🎉 All checkpoints processed successfully!"
        exit 0
    else
        print_warning "⚠️  Some checkpoints failed. Check the output above for details."
        exit 1
    fi
}

# Run main function
main "$@" 