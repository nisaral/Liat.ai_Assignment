# Player Re-Identification Assignment - Setup Guide
*Liat.ai Internship Task*

## Overview

This assignment involves implementing player re-identification systems for sports footage analysis. You'll work with computer vision techniques to track and identify players across different scenarios:

- **Option 1**: Cross-camera player mapping between broadcast and tactical camera feeds
- **Option 2**: Single-feed re-identification with temporal consistency

## Problem Statement

### Option 1: Cross-Camera Player Mapping
- **Objective**: Map players between two different camera angles (`broadcast.mp4` and `tacticam.mp4`) of the same gameplay
- **Challenge**: Maintain consistent player IDs across different camera perspectives  
- **Approach**: Use visual, spatial, and temporal features to establish player correspondence

### Option 2: Re-Identification in Single Feed
- **Objective**: Track players in a single video (`15sec_input_720p.mp4`) and maintain IDs when they go out of frame and return
- **Challenge**: Re-assign the same ID to players who temporarily disappear and reappear

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM, 16GB recommended
- 5GB+ free storage space

### Installation

1. **Clone/Download the assignment materials**
```bash
# Create project directory
mkdir player_reid_assignment
cd player_reid_assignment
```

2. **Create virtual environment**
```bash
# Using conda (recommended)
conda create -n player_reid python=3.9
conda activate player_reid

# Or using venv
python -m venv player_reid_env
# Windows: player_reid_env\Scripts\activate
# Linux/Mac: source player_reid_env/bin/activate
```

3. **Install required packages**
```bash
pip install ultralytics opencv-python torch torchvision numpy scipy scikit-learn matplotlib
pip install gdown tqdm seaborn pillow  # Optional but recommended
```

### Verify Installation
```python
import torch
import cv2
from ultralytics import YOLO
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"OpenCV version: {cv2.__version__}")
```

## File Structure

```
player_reid_assignment/
├── model.pt                      # Pre-trained YOLO model
├── videos/
│   ├── broadcast.mp4            # Broadcast camera feed
│   ├── tacticam.mp4             # Tactical camera feed
│   └── 15sec_input_720p.mp4     # Single feed input
├── src/
│   ├── cross_camera_mapping.py  # Option 1 implementation
│   ├── single_feed_reid.py      # Option 2 implementation
│   └── utils/
│       ├── tracking.py          # Tracking utilities
│       ├── features.py          # Feature extraction
│       └── visualization.py     # Result visualization
├── results/                     # Output directory
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Download Assets

### Model and Videos
```python
import gdown
import os

# Create directories
os.makedirs('videos', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Download model (replace with actual file ID)
model_url = "https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOqZNAMScrePVcMD/view"
gdown.download(model_url, 'model.pt', quiet=False, fuzzy=True)

# Download videos from assignment materials
# Update with actual video URLs
videos_folder_url = "https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6I8WknXd4Cv2c3vZTP"
# Download individual video files to videos/ directory
```

## Option 1: Cross-Camera Player Mapping

### Implementation Overview
This task requires establishing correspondence between players across two camera feeds viewing the same scene from different angles.

### Key Components
1. **Player Detection**: Use YOLO model to detect players in both feeds
2. **Feature Extraction**: Extract visual and spatial features for each player
3. **Temporal Alignment**: Synchronize frames between camera feeds
4. **Correspondence Matching**: Map players using similarity metrics

### Configuration
```python
# cross_camera_mapping.py configuration
CONFIG = {
    'MODEL_PATH': "model.pt",
    'BROADCAST_VIDEO': "videos/broadcast.mp4",
    'TACTICAM_VIDEO': "videos/tacticam.mp4",
    'OUTPUT_DIR': "results/cross_camera",
    'CONFIDENCE_THRESHOLD': 0.5,
    'IOU_THRESHOLD': 0.7,
    'FEATURE_DIM': 512,
    'SIMILARITY_THRESHOLD': 0.6
}
```

### Running Option 1
```bash
python src/cross_camera_mapping.py
```

### Expected Output
```
Processing broadcast video...
Detected 1250 player instances across 450 frames
Extracting features...

Processing tacticam video...  
Detected 1180 player instances across 450 frames
Extracting features...

Performing temporal alignment...
Computing player correspondences...

Cross-camera mapping results:
- Total broadcast players: 8
- Total tacticam players: 8  
- Successful mappings: 7
- Mapping confidence: 85.4%

Player Correspondences:
Broadcast Player 0 ↔ Tacticam Player 2 (confidence: 0.89)
Broadcast Player 1 ↔ Tacticam Player 0 (confidence: 0.92)
Broadcast Player 3 ↔ Tacticam Player 1 (confidence: 0.78)
...

Results saved to: results/cross_camera/
```

### Output Files
- `player_mappings.json`: Detailed mapping results
- `correspondence_matrix.png`: Visualization of player correspondences
- `broadcast_tracks.json`: Player tracks from broadcast feed
- `tacticam_tracks.json`: Player tracks from tactical feed

## Option 2: Single Feed Re-Identification

### Implementation Overview
Track players in a single video feed and maintain consistent IDs even when players temporarily leave the frame.

### Key Components
1. **Multi-Object Tracking**: Track detected players across frames
2. **Feature Bank**: Maintain appearance features for each tracked player
3. **Re-ID Matching**: Match reappearing players to existing tracks
4. **ID Management**: Handle track creation, deletion, and reassignment

### Configuration
```python
# single_feed_reid.py configuration
CONFIG = {
    'MODEL_PATH': "model.pt",
    'INPUT_VIDEO': "videos/15sec_input_720p.mp4",
    'OUTPUT_VIDEO': "results/tracked_output.mp4",
    'OUTPUT_DIR': "results/single_feed",
    'TRACK_BUFFER': 30,        # Frames to keep lost tracks
    'MATCH_THRESHOLD': 0.7,    # Re-ID similarity threshold
    'MIN_TRACK_LENGTH': 10,    # Minimum frames for valid track
    'FEATURE_UPDATE_RATE': 5   # Update features every N frames
}
```

### Running Option 2
```bash
python src/single_feed_reid.py
```

### Expected Output
```
Starting player re-identification...
Video info: 450 frames at 30.0 FPS (15.0 seconds)

Processing frames:
Frame 100/450 (22.2%) - Active tracks: 6, Total players: 6
Frame 200/450 (44.4%) - Active tracks: 5, Total players: 8  
Frame 300/450 (66.7%) - Active tracks: 7, Total players: 9
Frame 400/450 (88.9%) - Active tracks: 4, Total players: 9
Frame 450/450 (100.0%) - Processing complete

Re-identification Summary:
═══════════════════════════════════════
Total unique players tracked: 9
Total re-identification events: 12
Average track length: 156.3 frames
Tracking accuracy: 94.2%

Player Statistics:
Player 0: 234 detections, 15.0s duration, 2 re-ID events
Player 1: 189 detections, 12.1s duration, 1 re-ID event  
Player 2: 145 detections, 8.7s duration, 3 re-ID events
Player 3: 198 detections, 13.2s duration, 0 re-ID events
...

Output saved: results/tracked_output.mp4
Detailed logs: results/single_feed/tracking_log.json
```

### Output Files
- `tracked_output.mp4`: Video with player ID annotations
- `tracking_log.json`: Detailed tracking information
- `player_trajectories.png`: Visualization of player paths
- `reid_events.json`: Re-identification event details

## Dependencies

### Core Requirements
```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
```

### Optional Enhancements
```txt
gdown>=4.7.0          # Google Drive downloads
tqdm>=4.64.0           # Progress bars
seaborn>=0.11.0        # Enhanced visualizations
pillow>=9.0.0          # Image processing
tensorboard>=2.10.0    # Training visualization
```

## Hardware Requirements

### Minimum Configuration
- **CPU**: Intel i5 / AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional (CPU processing supported)

### Recommended Configuration  
- **CPU**: Intel i7/i9 / AMD Ryzen 7/9
- **RAM**: 16-32GB for large videos
- **GPU**: NVIDIA GTX 1060 / RTX 3060 or better
- **Storage**: SSD with 10GB+ free space

## Performance Optimization

### GPU Acceleration
```python
import torch
from ultralytics import YOLO

# Verify GPU setup
def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("GPU not available - using CPU")
        return False

# Initialize model with GPU
model = YOLO('model.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

### Memory Management
```python
import gc
import torch

def clear_memory():
    """Clear GPU/CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Process large videos in batches
def process_video_batched(video_path, batch_size=50):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for start_frame in range(0, frame_count, batch_size):
        # Process batch
        end_frame = min(start_frame + batch_size, frame_count)
        batch_results = process_frame_batch(cap, start_frame, end_frame)
        
        # Clear memory after each batch
        clear_memory()
    
    cap.release()
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```python
# Issue: FileNotFoundError: model.pt not found
# Solution:
import os
if not os.path.exists('model.pt'):
    print("Model file missing. Please download model.pt")
    # Download using gdown or manual download

# Issue: Model format incompatible
# Solution: Verify YOLO version compatibility
from ultralytics import __version__
print(f"Ultralytics version: {__version__}")
```

#### 2. CUDA Memory Issues
```python
# Issue: RuntimeError: CUDA out of memory
# Solutions:
torch.cuda.empty_cache()           # Clear cache
batch_size = 1                     # Reduce batch size
imgsz = 640                        # Reduce input size (default: 1280)

# Process with reduced image size
results = model(frame, imgsz=640, device='cuda')
```

#### 3. Video Codec Problems
```bash
# Issue: Cannot read video file
# Solution: Install FFmpeg
# Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg
# Windows: Download from https://ffmpeg.org/
# Mac: brew install ffmpeg

# Convert video format if needed
ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4
```

#### 4. Slow Processing
```python
# Optimization strategies:
# 1. Use GPU acceleration
model.to('cuda')

# 2. Reduce input resolution
results = model(frame, imgsz=640)  # Instead of 1280

# 3. Skip frames for faster processing
frame_skip = 2  # Process every 2nd frame
if frame_idx % frame_skip == 0:
    results = model(frame)

# 4. Use half precision (if supported)
model.half()  # Use FP16 instead of FP32
```

## Validation and Testing

### 1. Environment Validation
```python
def validate_environment():
    """Check if environment is properly set up"""
    checks = {
        'Python': None,
        'PyTorch': None,
        'CUDA': None,
        'OpenCV': None,
        'Model': None
    }
    
    # Check Python version
    import sys
    if sys.version_info >= (3, 8):
        checks['Python'] = '✓'
    else:
        checks['Python'] = '✗ (requires 3.8+)'
    
    # Check PyTorch
    try:
        import torch
        checks['PyTorch'] = f"✓ ({torch.__version__})"
    except:
        checks['PyTorch'] = '✗'
    
    # Check CUDA
    if torch.cuda.is_available():
        checks['CUDA'] = f"✓ ({torch.cuda.get_device_name(0)})"
    else:
        checks['CUDA'] = '○ (CPU only)'
    
    # Check OpenCV
    try:
        import cv2
        checks['OpenCV'] = f"✓ ({cv2.__version__})"
    except:
        checks['OpenCV'] = '✗'
    
    # Check model file
    if os.path.exists('model.pt'):
        checks['Model'] = '✓'
    else:
        checks['Model'] = '✗ (model.pt not found)'
    
    print("Environment Validation:")
    print("=" * 30)
    for component, status in checks.items():
        print(f"{component:10}: {status}")

# Run validation
validate_environment()
```

### 2. Model Testing
```python
def test_model():
    """Test model inference on sample data"""
    from ultralytics import YOLO
    import numpy as np
    
    try:
        model = YOLO('model.pt')
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image)
        
        print(f"Model test: ✓ (detected {len(results[0].boxes)} objects)")
        return True
    except Exception as e:
        print(f"Model test: ✗ ({str(e)})")
        return False

test_model()
```

### 3. Video Processing Test
```python
def test_video_processing():
    """Test video file accessibility"""
    test_videos = [
        'videos/broadcast.mp4',
        'videos/tacticam.mp4', 
        'videos/15sec_input_720p.mp4'
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✓ {video_path}: {frame_count} frames @ {fps:.1f} FPS")
                cap.release()
            else:
                print(f"✗ {video_path}: Cannot open video")
        else:
            print(f"✗ {video_path}: File not found")

test_video_processing()
```

## Evaluation Metrics

### Option 1: Cross-Camera Mapping
- **Mapping Accuracy**: Percentage of correctly mapped players
- **Correspondence Confidence**: Average similarity score for mappings
- **Temporal Consistency**: Stability of mappings across time

### Option 2: Single Feed Re-ID
- **Tracking Accuracy**: Percentage of correctly maintained IDs
- **Re-ID Success Rate**: Percentage of successful re-identifications
- **Identity Switches**: Number of incorrect ID assignments
- **Track Fragmentation**: Average number of fragments per true track

## Submission Guidelines

### Deliverables
1. **Source Code**: Complete implementation with comments
2. **Results**: Output videos and JSON files
3. **Report**: Brief analysis of approach and results
4. **Demo**: Working demonstration of both options

### Code Structure
- Clean, readable, and well-documented code
- Modular design with reusable components  
- Error handling and input validation
- Configuration management

### Performance Benchmarks
Include processing speed and accuracy metrics:
- Frames per second (FPS) processing rate
- Memory usage statistics
- GPU utilization (if applicable)
- Accuracy/precision metrics

## Additional Resources

### Documentation
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

### Useful Papers
- "Deep Learning for Multi-Object Tracking: A Survey"
- "Person Re-identification: Past, Present and Future"
- "Multiple Object Tracking: A Literature Review"

---

**Good luck with your implementation!** 

For questions or issues, please contact the Liat.ai internship supervisor.
