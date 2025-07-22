#!/bin/bash

# Script to extract video segments using ffmpeg
# Usage: ./extract_video_segments.sh <input_video> <output_dir> <interval1> <interval2> ...
# Example: ./extract_video_segments.sh video.mp4 output/ "2-13" "15-25" "28-41"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_video> <output_dir> <interval1> [interval2] [interval3] ..."
    echo "Example: $0 video.mp4 output/ \"2-13\" \"15-25\" \"28-41\""
    echo "Intervals format: \"start-end\" (in seconds)"
    exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
shift 2  # Remove first two arguments, leaving only intervals

# Check if input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video '$INPUT_VIDEO' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the base name of the input video (without path and extension)
BASE_NAME=$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')

echo "Processing video: $INPUT_VIDEO"
echo "Output directory: $OUTPUT_DIR"
echo "Base name: $BASE_NAME"
echo ""

# Counter for segment numbering
SEGMENT_NUM=1

# Process each interval
for INTERVAL in "$@"; do
    # Parse start and end times from interval (format: "start-end")
    if [[ $INTERVAL =~ ^([0-9]+)-([0-9]+)$ ]]; then
        START_TIME=${BASH_REMATCH[1]}
        END_TIME=${BASH_REMATCH[2]}
        
        # Calculate duration
        DURATION=$((END_TIME - START_TIME))
        
        # Create output filename
        OUTPUT_FILE="${OUTPUT_DIR}/${BASE_NAME}_segment${SEGMENT_NUM}_${START_TIME}s-${END_TIME}s.mp4"
        
        echo "Extracting segment $SEGMENT_NUM: ${START_TIME}s to ${END_TIME}s (duration: ${DURATION}s)"
        echo "Output: $OUTPUT_FILE"
        
        # Extract segment using ffmpeg
        ffmpeg -i "$INPUT_VIDEO" \
               -ss "$START_TIME" \
               -t "$DURATION" \
               -c copy \
               -avoid_negative_ts make_zero \
               "$OUTPUT_FILE" \
               -y  # Overwrite output files without asking
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully extracted segment $SEGMENT_NUM"
        else
            echo "✗ Failed to extract segment $SEGMENT_NUM"
        fi
        
        echo ""
        SEGMENT_NUM=$((SEGMENT_NUM + 1))
    else
        echo "Warning: Invalid interval format '$INTERVAL'. Expected format: 'start-end' (e.g., '2-13')"
        echo ""
    fi
done

echo "All segments processed!"
echo "Output files saved in: $OUTPUT_DIR"
