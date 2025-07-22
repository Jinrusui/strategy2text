#!/bin/bash
# Array of original video paths
ORIGINAL_VIDEOS=(
    "/mnt/e/Projects/strategy2text/record_seed42/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed76/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed100/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed420/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed760/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
) to extract 30-60s segments from all original videos with different seeds
# 从所有不同seed的原始视频中提取30-60秒片段

echo "Starting batch extraction of 30-60s segments from original videos..."
echo "开始批量提取原始视频的30-60秒片段..."

# Create output directory for all segments
OUTPUT_BASE_DIR="/mnt/e/Projects/strategy2text/original_video_segments_30-60s"
mkdir -p "$OUTPUT_BASE_DIR"

# Array of original video paths
ORIGINAL_VIDEOS=(
    "/mnt/e/Projects/strategy2text/record_seed42/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed76/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed100/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed420/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
    "/mnt/e/Projects/strategy2text/record_seed760/original/final-model-dqn-BreakoutNoFrameskip-v4_original-step-0-to-step-2000.mp4"
)

# Extract seed numbers for naming
SEEDS=("seed42" "seed76" "seed100" "seed420" "seed760")

# Time interval: 30-60 seconds (30 seconds duration)
START_TIME=30
END_TIME=60
DURATION=$((END_TIME - START_TIME))

echo "Extracting ${START_TIME}s to ${END_TIME}s (duration: ${DURATION}s) from each video"
echo "从每个视频提取${START_TIME}秒到${END_TIME}秒（时长：${DURATION}秒）"
echo ""

# Counter for processing
count=0

# Process each video
for i in "${!ORIGINAL_VIDEOS[@]}"; do
    VIDEO_PATH="${ORIGINAL_VIDEOS[$i]}"
    SEED="${SEEDS[$i]}"
    
    # Check if video file exists
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "⚠️  Warning: Video not found: $VIDEO_PATH"
        echo "⚠️  警告：视频文件未找到：$VIDEO_PATH"
        continue
    fi
    
    # Create output filename
    OUTPUT_FILE="${OUTPUT_BASE_DIR}/BreakoutNoFrameskip-v4_dqn_${SEED}_original_30s-60s.mp4"
    
    echo "Processing ${SEED}..."
    echo "正在处理 ${SEED}..."
    echo "Input:  $VIDEO_PATH"
    echo "Output: $OUTPUT_FILE"
    
    # Extract segment using ffmpeg
    ffmpeg -i "$VIDEO_PATH" \
           -ss "$START_TIME" \
           -t "$DURATION" \
           -c copy \
           -avoid_negative_ts make_zero \
           "$OUTPUT_FILE" \
           -y
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully extracted segment from ${SEED}"
        echo "✅ 成功提取 ${SEED} 的片段"
        count=$((count + 1))
    else
        echo "❌ Failed to extract segment from ${SEED}"
        echo "❌ 提取 ${SEED} 片段失败"
    fi
    
    echo ""
done

echo "=================================================="
echo "Batch processing completed!"
echo "批处理完成！"
echo "Successfully processed: $count videos"
echo "成功处理: $count 个视频"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "输出目录: $OUTPUT_BASE_DIR"
echo "=================================================="

# List the created files
echo "Created files / 创建的文件:"
ls -la "$OUTPUT_BASE_DIR"
