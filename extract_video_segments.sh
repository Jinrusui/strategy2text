#!/bin/bash

# Script to extract video segments using ffmpeg with high accuracy
# Usage: ./extract_video_segments.sh <input_video> <output_dir> <interval1> <interval2> ...
# Example: ./extract_video_segments.sh video.mp4 output/ "2.5-13.2" "15-25" "1:28-1:41.5"
# Supported time formats: seconds (2.5), MM:SS (1:30), HH:MM:SS (1:30:45), with decimal precision

if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_video> <output_dir> <interval1> [interval2] [interval3] ..."
    echo "Example: $0 video.mp4 output/ \"2.5-13.2\" \"15-25\" \"1:28-1:41.5\""
    echo "Time formats supported:"
    echo "  - Seconds: 2.5, 13, 45.75"
    echo "  - MM:SS: 1:30, 2:15.5"
    echo "  - HH:MM:SS: 1:30:45, 0:02:15.25"
    echo "Intervals format: \"start-end\""
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

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed or not in PATH!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Get the base name of the input video (without path and extension)
BASE_NAME=$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')

# Function to convert time format to seconds
time_to_seconds() {
    local time_str="$1"
    
    # Check if it's already in seconds (decimal number)
    if [[ $time_str =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "$time_str"
        return
    fi
    
    # Check if it's in MM:SS or MM:SS.decimal format
    if [[ $time_str =~ ^([0-9]+):([0-9]+)(\.[0-9]+)?$ ]]; then
        local minutes=${BASH_REMATCH[1]}
        local seconds=${BASH_REMATCH[2]}
        local decimal=${BASH_REMATCH[3]:-}
        echo "$((minutes * 60 + seconds))$decimal"
        return
    fi
    
    # Check if it's in HH:MM:SS or HH:MM:SS.decimal format
    if [[ $time_str =~ ^([0-9]+):([0-9]+):([0-9]+)(\.[0-9]+)?$ ]]; then
        local hours=${BASH_REMATCH[1]}
        local minutes=${BASH_REMATCH[2]}
        local seconds=${BASH_REMATCH[3]}
        local decimal=${BASH_REMATCH[4]:-}
        echo "$((hours * 3600 + minutes * 60 + seconds))$decimal"
        return
    fi
    
    # Invalid format
    echo "INVALID"
}

# Function to get video duration
get_video_duration() {
    local video="$1"
    ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null
}

# Get video duration for validation
VIDEO_DURATION=$(get_video_duration "$INPUT_VIDEO")
if [ -z "$VIDEO_DURATION" ] || [ "$VIDEO_DURATION" = "N/A" ]; then
    echo "Warning: Could not determine video duration. Skipping duration validation."
    VIDEO_DURATION=""
fi

echo "Processing video: $INPUT_VIDEO"
if [ -n "$VIDEO_DURATION" ]; then
    echo "Video duration: ${VIDEO_DURATION}s"
fi
echo "Output directory: $OUTPUT_DIR"
echo "Base name: $BASE_NAME"
echo ""

# Counter for segment numbering
SEGMENT_NUM=1

# Process each interval
for INTERVAL in "$@"; do
    # Parse start and end times from interval (format: "start-end")
    if [[ $INTERVAL =~ ^(.+)-(.+)$ ]]; then
        START_TIME_STR=${BASH_REMATCH[1]}
        END_TIME_STR=${BASH_REMATCH[2]}
        
        # Convert times to seconds
        START_TIME=$(time_to_seconds "$START_TIME_STR")
        END_TIME=$(time_to_seconds "$END_TIME_STR")
        
        # Validate time formats
        if [ "$START_TIME" = "INVALID" ]; then
            echo "Error: Invalid start time format '$START_TIME_STR' in interval '$INTERVAL'"
            echo "Skipping this interval."
            echo ""
            continue
        fi
        
        if [ "$END_TIME" = "INVALID" ]; then
            echo "Error: Invalid end time format '$END_TIME_STR' in interval '$INTERVAL'"
            echo "Skipping this interval."
            echo ""
            continue
        fi
        
        # Validate time range
        if (( $(echo "$START_TIME >= $END_TIME" | bc -l) )); then
            echo "Error: Start time ($START_TIME_STR) must be less than end time ($END_TIME_STR)"
            echo "Skipping this interval."
            echo ""
            continue
        fi
        
        # Validate against video duration if available
        if [ -n "$VIDEO_DURATION" ]; then
            if (( $(echo "$START_TIME >= $VIDEO_DURATION" | bc -l) )); then
                echo "Error: Start time ($START_TIME_STR) exceeds video duration (${VIDEO_DURATION}s)"
                echo "Skipping this interval."
                echo ""
                continue
            fi
            
            if (( $(echo "$END_TIME > $VIDEO_DURATION" | bc -l) )); then
                echo "Warning: End time ($END_TIME_STR) exceeds video duration (${VIDEO_DURATION}s)"
                echo "Adjusting end time to video duration."
                END_TIME="$VIDEO_DURATION"
                END_TIME_STR="${VIDEO_DURATION}s"
            fi
        fi
        
        # Calculate duration
        DURATION=$(echo "$END_TIME - $START_TIME" | bc -l)
        
        # Create output filename
        OUTPUT_FILE="${OUTPUT_DIR}/${BASE_NAME}_segment${SEGMENT_NUM}_${START_TIME_STR//[:.]/}s-${END_TIME_STR//[:.]/}s.mp4"
        
        echo "Extracting segment $SEGMENT_NUM: ${START_TIME_STR} to ${END_TIME_STR} (duration: ${DURATION}s)"
        echo "Output: $OUTPUT_FILE"
        
        # Extract segment using ffmpeg with high accuracy
        # Use -ss before -i for faster seeking, then re-encode for precision
        ffmpeg -ss "$START_TIME" \
               -i "$INPUT_VIDEO" \
               -t "$DURATION" \
               -c:v libx264 \
               -c:a aac \
               -preset fast \
               -crf 23 \
               -avoid_negative_ts make_zero \
               -movflags +faststart \
               "$OUTPUT_FILE" \
               -y 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully extracted segment $SEGMENT_NUM"
            
            # Verify output file was created and has reasonable size
            if [ -f "$OUTPUT_FILE" ]; then
                FILE_SIZE=$(stat -f%z "$OUTPUT_FILE" 2>/dev/null || stat -c%s "$OUTPUT_FILE" 2>/dev/null)
                if [ "$FILE_SIZE" -gt 1000 ]; then  # At least 1KB
                    echo "  File size: $(numfmt --to=iec --suffix=B $FILE_SIZE)"
                else
                    echo "  Warning: Output file seems unusually small ($FILE_SIZE bytes)"
                fi
            fi
        else
            echo "✗ Failed to extract segment $SEGMENT_NUM"
            echo "  Check if the time range is valid and within video bounds"
        fi
        
        echo ""
        SEGMENT_NUM=$((SEGMENT_NUM + 1))
    else
        echo "Warning: Invalid interval format '$INTERVAL'"
        echo "Expected format: 'start-end' (e.g., '2.5-13.2', '1:30-2:45', '0:01:15-0:02:30.5')"
        echo ""
    fi
done

echo "All segments processed!"
echo "Output files saved in: $OUTPUT_DIR"

# Summary
TOTAL_SEGMENTS=$((SEGMENT_NUM - 1))
SUCCESSFUL_SEGMENTS=$(find "$OUTPUT_DIR" -name "${BASE_NAME}_segment*.mp4" -size +1000c 2>/dev/null | wc -l)
echo "Summary: $SUCCESSFUL_SEGMENTS out of $TOTAL_SEGMENTS segments extracted successfully"
