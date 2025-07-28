# HVA-X Phase Scripts - Output Directory Support

所有的 HVA-X 阶段脚本现在都支持 `--output-dir` 参数，允许您指定输出文件的保存目录。

## 支持的脚本

- `run_phase1_only.py` - 轨迹采样和分层
- `run_phase2a_only.py` - 事件检测
- `run_phase2b_only.py` - 引导分析
- `run_phase3_only.py` - 元综合

## 使用方法

### 基本语法
```bash
python run_phase[N]_only.py [其他参数] --output-dir [目录路径]
```

### 示例

#### Phase 1 - 轨迹采样
```bash
# 保存到默认目录（当前目录）
python run_phase1_only.py --video-dir hva_videos

# 保存到指定目录
python run_phase1_only.py --video-dir hva_videos --output-dir results/phase1

# 使用自定义前缀和目录
python run_phase1_only.py --video-dir hva_videos --output-prefix my_sampling --output-dir outputs/
```

#### Phase 2A - 事件检测
```bash
# 从Phase 1结果进行事件检测
python run_phase2a_only.py --phase1-file phase1_sampling_20240101_120000.json --output-dir results/phase2a

# 直接模式处理所有视频
python run_phase2a_only.py --video-dir video_clips_30s --direct-mode --output-dir results/events/
```

#### Phase 2B - 引导分析
```bash
# 使用Phase 2A结果进行引导分析
python run_phase2b_only.py --phase2a-file phase2a_events_20240101_120000.json --output-dir results/phase2b/
```

#### Phase 3 - 元综合
```bash
# 生成最终报告
python run_phase3_only.py --phase2b-file phase2b_analysis_20240101_120000.json --output-dir results/final/ --save-report
```

## 目录结构建议

推荐使用以下目录结构来组织输出文件：

```
results/
├── phase1/           # 轨迹采样结果
├── phase2a/          # 事件检测结果  
├── phase2b/          # 引导分析结果
├── phase3/           # 元综合结果
└── reports/          # 最终报告
```

### 创建目录结构
```bash
mkdir -p results/{phase1,phase2a,phase2b,phase3,reports}
```

### 完整流水线示例
```bash
# Phase 1: 轨迹采样
python run_phase1_only.py --video-dir hva_videos --output-dir results/phase1/

# Phase 2A: 事件检测
python run_phase2a_only.py --phase1-file results/phase1/phase1_sampling_*.json --output-dir results/phase2a/

# Phase 2B: 引导分析
python run_phase2b_only.py --phase2a-file results/phase2a/phase2a_events_*.json --output-dir results/phase2b/

# Phase 3: 元综合
python run_phase3_only.py --phase2b-file results/phase2b/phase2b_analysis_*.json --output-dir results/phase3/ --save-report
```

## 注意事项

1. **目录自动创建**: 如果指定的输出目录不存在，脚本会自动创建它
2. **相对路径**: 可以使用相对路径（如 `results/phase1`）或绝对路径
3. **默认行为**: 如果不指定 `--output-dir`，文件将保存到当前目录
4. **文件命名**: 输出文件名格式为 `{prefix}_{timestamp}.json`，其中 timestamp 为 `YYYYMMDD_HHMMSS` 格式

## 参数说明

- `--output-dir`: 指定输出文件保存的目录（默认：当前目录 "."）
- `--output-prefix`: 指定输出文件的前缀（每个阶段有不同的默认值）

通过这些参数，您可以灵活地组织和管理 HVA-X 分析流程的所有输出文件。
