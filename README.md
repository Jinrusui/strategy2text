# Strategy2Text: Dual-Level RL Agent Video Analysis with Gemini AI

This project uses Google's Gemini AI to analyze reinforcement learning agent videos with **two specialized analyzers**:
- **BehaviorAnalyzer**: Focuses on low-level, moment-to-moment behaviors and actions
- **BreakoutStrategyAnalyzer**: Focuses on high-level strategic thinking and planning

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Google API Key

Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and set it as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 3. Run the Comprehensive Analysis

```bash
python analyze_behavior_and_strategy.py path/to/your/video.mp4
```

## 📋 Usage Examples

### Comprehensive Analysis (Both Analyzers)

```python
from src.gemini_analysis import BehaviorAnalyzer, BreakoutStrategyAnalyzer

# Initialize both analyzers
behavior_analyzer = BehaviorAnalyzer()
strategy_analyzer = BreakoutStrategyAnalyzer()

# Analyze low-level behaviors
behavior_result = behavior_analyzer.analyze_behavior("path/to/video.mp4")
print("Behavior Summary:", behavior_result["behavior_summary"]["summary"])

# Analyze high-level strategy
strategy_result = strategy_analyzer.analyze_breakout_strategy("path/to/video.mp4")
print("Strategy Analysis:", strategy_result["strategy_summary"])
```

### Low-Level Behavior Analysis Only

```python
from src.gemini_analysis import BehaviorAnalyzer

# Initialize the behavior analyzer
analyzer = BehaviorAnalyzer()

# Analyze moment-to-moment behaviors
analysis = analyzer.analyze_behavior("path/to/video.mp4")

# Get structured behavior breakdown
behavior_summary = analysis["behavior_summary"]
print("Immediate Actions:", behavior_summary["immediate_actions"])
print("Movement Patterns:", behavior_summary["movement_patterns"])
print("Decision Making:", behavior_summary["decision_making"])
print("Technical Execution:", behavior_summary["technical_execution"])
```

### High-Level Strategy Analysis Only

```python
from src.gemini_analysis import BreakoutStrategyAnalyzer

# Initialize the strategy analyzer
analyzer = BreakoutStrategyAnalyzer()

# Analyze strategic thinking
analysis = analyzer.analyze_breakout_strategy("path/to/video.mp4")

# Get structured strategy breakdown
strategy_summary = analysis["strategy_summary"]
print("Overall Strategy:", strategy_summary["overall_strategy"])
print("Strategic Brick Destruction:", strategy_summary["strategic_brick_destruction"])
print("Game State Management:", strategy_summary["game_state_management"])
```

### Batch Analysis

```python
from src.gemini_analysis import BehaviorAnalyzer, BreakoutStrategyAnalyzer
from src.video_processing import VideoLoader

# Find all videos in a directory
loader = VideoLoader()
video_paths = loader.find_videos("videos/")

# Batch analyze behaviors
behavior_analyzer = BehaviorAnalyzer()
behavior_results = behavior_analyzer.batch_analyze_behaviors(video_paths)

# Batch analyze strategies
strategy_analyzer = BreakoutStrategyAnalyzer()
strategy_results = strategy_analyzer.batch_analyze_breakout_strategies(video_paths)
```

## 🏗️ Architecture

The system consists of **dual-level analysis** with two specialized analyzers:

### `src/gemini_analysis/`
- **`GeminiClient`**: Low-level interface to Gemini AI with support for both upload and direct video analysis
- **`BehaviorAnalyzer`**: Specialized for low-level behavior analysis (moment-to-moment actions)
- **`BreakoutStrategyAnalyzer`**: Specialized for high-level strategic analysis (planning and decision-making)

### `src/video_processing/`
- **`VideoLoader`**: Utilities for finding, validating, and organizing video files

## 📊 Analysis Output

### Behavior Analysis Output
Focuses on **low-level, immediate behaviors**:
- **Immediate Actions & Reactions**: Frame-by-frame action sequences and reaction times
- **Movement Patterns**: Speed, acceleration, and direction change patterns
- **Decision-Making Process**: Speed and evidence of exploration vs exploitation
- **Technical Execution**: Accuracy, consistency, and error patterns
- **3-Sentence Summary**: Concise behavioral characteristics

### Strategy Analysis Output
Focuses on **high-level strategic thinking**:
- **Overall Strategy & Approach**: Long-term planning and risk assessment
- **Strategic Brick Destruction**: Prioritization and planning for efficient clearing
- **Game State Management**: Adaptation to different scenarios and configurations
- **Learning & Strategic Evolution**: Evidence of strategic improvement and pattern recognition
- **Performance & Strategic Effectiveness**: Overall strategic success and optimization

## 🎯 Key Features

### Two Analysis Modes
- **Low-Level Behavior Analysis**: What the agent is doing moment-to-moment
- **High-Level Strategy Analysis**: How the agent plans and makes strategic decisions

### Video Processing
- ✅ Support for videos <20MB using direct inline analysis
- ✅ Support for larger videos using upload-based analysis
- ✅ Automatic file size detection and method selection
- ✅ Configurable FPS settings for video processing

### Analysis Capabilities
- ✅ Single video analysis with detailed breakdowns
- ✅ Batch analysis across multiple videos

- ✅ Structured JSON output for further processing
- ✅ Error handling and retry logic
- ✅ Context manager support for proper cleanup

## 📁 File Structure

```
strategy2text/
├── src/
│   ├── gemini_analysis/
│   │   ├── __init__.py
│   │   ├── gemini_client.py         # Core Gemini AI interface
│   │   ├── behavior_analyzer.py     # Low-level behavior analysis
│   │   └── strategy_analyzer.py     # High-level strategic analysis
│   └── video_processing/
│       ├── __init__.py
│       └── video_loader.py          # Video file utilities
├── analyze_behavior_and_strategy.py # Comprehensive analysis example
├── analyze_videos.py                # General example usage script
├── analyze_breakout.py              # Specific Breakout analysis script
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google API key (required)

### Supported Video Formats
- MP4, AVI, MOV, MKV, WebM

### Model Configuration
- **Behavior Analyzer**: Uses `gemini-2.5-flash-preview-05-20` for detailed frame analysis
- **Strategy Analyzer**: Uses `gemini-2.0-flash` for strategic reasoning

## 🎮 Analysis Focus Areas

### Behavior Analysis (Low-Level)
- **Immediate Actions**: Frame-by-frame action sequences
- **Movement Precision**: Micro-adjustments and fine-grained control
- **Reaction Times**: Response speed to environmental changes
- **Technical Execution**: Accuracy and consistency in similar situations

### Strategy Analysis (High-Level)
- **Strategic Planning**: Long-term game plans and risk assessment
- **Pattern Recognition**: Strategic adaptation based on game feedback
- **Optimization**: Efficiency of strategic choices and resource allocation
- **Learning Evolution**: Evidence of strategic improvement over time

## 🔧 Advanced Usage

### Custom Analysis with Direct Video Processing

```python
from src.gemini_analysis import GeminiClient
from google.genai import types

# For videos <20MB - direct processing approach
client = GeminiClient(model="gemini-2.5-flash-preview-05-20")

# Read video bytes
with open("path/to/video.mp4", 'rb') as f:
    video_bytes = f.read()

# Custom analysis with direct approach
response = client.client.models.generate_content(
    model='gemini-2.5-flash-preview-05-20',
    contents=types.Content(
        parts=[
            types.Part(
                inline_data=types.Blob(
                    data=video_bytes,
                    mime_type='video/mp4'
                ),
                video_metadata=types.VideoMetadata(fps=5)
            ),
            types.Part(text='Your custom analysis prompt here')
        ]
    )
)
```

## 🤝 Contributing

Feel free to extend the system with:
- Enhanced behavior analysis prompts
- Additional strategic analysis dimensions
- Support for other game environments
- Performance metrics extraction
- Integration with RL training frameworks

## 📝 Notes

- **Dual Analysis**: The system provides both low-level behavioral and high-level strategic insights
- **Video Size Optimization**: Automatically chooses the best analysis method based on file size
- **Specialized Prompts**: Each analyzer uses carefully crafted prompts for their specific focus area
- **Structured Output**: All results are provided in structured JSON format for easy integration
- **Minimal Dependencies**: Uses only essential libraries to keep the system lightweight 