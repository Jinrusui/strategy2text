#!/bin/bash

# Script to extract frames from video clips, saving one image every 3 frames
# 从视频片段中提取帧，每3帧保存一张图片

echo "Starting frame extraction from video clips..."
echo "开始从视频片段中提取帧..."

# Input and output directories
INPUT_DIR="/mnt/e/Projects/strategy2text/video_clips_30s"
OUTPUT_BASE_DIR="/mnt/e/Projects/strategy2text/video_clips_30s_frames"

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' not found!"
    echo "错误：输入目录 '$INPUT_DIR' 未找到！"
    exit 1
fi

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_BASE_DIR"
echo ""

# Counter for processed videos
processed_count=0
total_frames=0

# Process each video file in the input directory
for video_file in "$INPUT_DIR"/*.mp4; do
    # Check if any mp4 files exist
    if [ ! -f "$video_file" ]; then
        echo "No MP4 files found in $INPUT_DIR"
        echo "在 $INPUT_DIR 中未找到MP4文件"
        continue
    fi
    
    # Get the base name without extension
    base_name=$(basename "$video_file" .mp4)
    
    # Create output directory for this video
    output_dir="$OUTPUT_BASE_DIR/${base_name}_frames"
    mkdir -p "$output_dir"
    
    echo "Processing: $base_name"
    echo "正在处理: $base_name"
    echo "Input:  $video_file"
    echo "Output: $output_dir"
    
    # Extract frames using ffmpeg
    # -vf "select='not(mod(n,3))'" selects every 3rd frame (0, 3, 6, 9, ...)
    # -vsync vfr ensures variable frame rate to match the selection
    # -q:v 2 sets high quality for output images
    ffmpeg -i "$video_file" \
           -vf "select='not(mod(n,3))'" \
           -vsync vfr \
           -q:v 2 \
           "$output_dir/frame_%04d.png" \
           -y
    
    if [ $? -eq 0 ]; then
        # Count the number of extracted frames
        frame_count=$(ls -1 "$output_dir"/frame_*.png 2>/dev/null | wc -l)
        echo "✅ Successfully extracted $frame_count frames from $base_name"
        echo "✅ 成功从 $base_name 提取了 $frame_count 帧"
        
        processed_count=$((processed_count + 1))
        total_frames=$((total_frames + frame_count))
    else
        echo "❌ Failed to extract frames from $base_name"
        echo "❌ 从 $base_name 提取帧失败"
    fi
    
    echo ""
done

echo "=================================================="
echo "Frame extraction completed!"
echo "帧提取完成！"
echo "Videos processed: $processed_count"
echo "Total frames extracted: $total_frames"
echo "处理的视频数: $processed_count"
echo "提取的总帧数: $total_frames"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "输出目录: $OUTPUT_BASE_DIR"
echo "=================================================="

# Show directory structure
echo ""
echo "Directory structure / 目录结构:"
for dir in "$OUTPUT_BASE_DIR"/*_frames; do
    if [ -d "$dir" ]; then
        dir_name=$(basename "$dir")
        frame_count=$(ls -1 "$dir"/frame_*.png 2>/dev/null | wc -l)
        echo "  $dir_name: $frame_count frames"
    fi
done
