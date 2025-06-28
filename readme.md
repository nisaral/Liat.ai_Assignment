# Player Re-Identification Assignment - Setup Guide
*Liat.ai Internship Task*

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Setup Instructions](#setup-instructions)
- [Environment Setup](#environment-setup)
- [File Structure](#file-structure)
- [Current Approach](#current-approach)
- [Implementation Options](#implementation-options)
- [Alternative Approaches](#alternative-approaches)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Validation and Testing](#validation-and-testing)
- [Evaluation Metrics](#evaluation-metrics)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Potential Improvements](#potential-improvements)
- [Submission Guidelines](#submission-guidelines)

## Overview

This assignment implements player tracking and re-identification systems for sports footage analysis using the YOLO (You Only Look Once) model from the Ultralytics library. The system extracts player features, such as color histograms and bounding box positions, to track and re-identify players across video frames.

The provided Jupyter notebook (`liat_ai_Task (1).ipynb`) implements a solution that:

- Uses the YOLO model for object detection to identify players in video frames
- Extracts color histogram features from detected player bounding boxes for re-identification
- Tracks players across frames using feature similarity (cosine similarity) and the Hungarian algorithm for optimal assignment
- Visualizes the results with bounding boxes and player IDs

The system is designed to run on a GPU-enabled environment (e.g., Google Colab with a T4 GPU) and leverages libraries like OpenCV, PyTorch, and scikit-learn for processing and analysis.

## Problem Statement

You'll work with computer vision techniques to track and identify players across different scenarios:

### Option 1: Cross-Camera Player Mapping
- **Objective**: Map players between two different camera angles (`broadcast.mp4` and `tacticam.mp4`) of the same gameplay
- **Challenge**: Maintain consistent player IDs across different camera perspectives  
- **Approach**: Use visual, spatial, and temporal features to establish player correspondence

### Option 2: Re-Identification in Single Feed
- **Objective**: Track players in a single video (`15sec_input_720p.mp4`) and maintain IDs when they go out of frame and return
- **Challenge**: Re-assign the same ID to players who temporarily disappear and reappear

## Setup Instructions

To run the notebook, follow these steps:

### Prerequisites
- Python 3.11 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM, 16GB recommended
- 5GB+ free storage space

### 1. Clone the Repository
```bash
# Create project directory
mkdir player_reid_assignment
cd player_reid_assignment

# Clone repository (if applicable)
git clone <repository-url>
cd <repository-directory>
```

### 2. Set Up Python Environment
```bash
# Ensure Python 3.11 or later is installed
python --version

# Using conda (recommended)
conda create -n player_reid python=3.11
conda activate player_reid

# Or using venv
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install required Python packages
pip install ultralytics opencv-python torch torchvision numpy scipy scikit-learn matplotlib
pip install gdown tqdm seaborn pillow  # Optional but recommended
```

### 4. Install NVIDIA CUDA and cuDNN (GPU Support)
Ensure CUDA 12.4 and cuDNN 9.1 are installed for GPU acceleration with PyTorch.

Verify GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 5. Prepare Input Data
- Ensure the input video or image sequence is available in the working directory
- Update file paths in the notebook if necessary

### 6. Run the Notebook
```bash
# Open the notebook in Jupyter
jupyter notebook liat_ai_Task\ \(1\).ipynb

# Execute the cells sequentially to:
# - Install dependencies
# - Load the model
# - Process the video
```

## Environment Setup

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
├── liat_ai_Task (1).ipynb        # Main implementation notebook
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

## Current Approach

The current implementation in the notebook uses the following components:

### 1. Object Detection with YOLO
- **Model**: Ultralytics YOLOv8 (pre-trained model, e.g., yolov8n.pt)
- **Purpose**: Detects players in each video frame by generating bounding boxes
- **Implementation**: 
  - The YOLO model is loaded using the `ultralytics.YOLO` class
  - Frames are processed to detect objects classified as "person" (or a custom class if fine-tuned)
  - Bounding boxes are extracted for further processing

### 2. Feature Extraction
- **Class**: `PlayerFeatureExtractor`
- **Features**:
  - **Color Histogram**: Extracts HSV color histograms from player bounding boxes (32 bins per channel, normalized)
  - **Position Features**: Normalizes bounding box coordinates (x1, y1, x2, y2) relative to frame dimensions
- **Purpose**: Generates feature vectors for each detected player to enable re-identification across frames

### 3. Player Tracking and Re-Identification
- **Method**: Combines feature similarity with the Hungarian algorithm for tracking
- **Similarity Metric**: Cosine similarity is computed between color histogram features of detected players in consecutive frames
- **Assignment**: The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) assigns player IDs by minimizing the cost matrix based on feature dissimilarity
- **Tracking Logic**: Maintains a deque (double-ended queue) to store player features and IDs, ensuring continuity across frames

### 4. Visualization
- **Tools**: Matplotlib and OpenCV
- **Output**: Frames with annotated bounding boxes and player IDs, saved as images or a video sequence

### 5. Rolling Gradients
The current approach does not explicitly implement rolling gradients, but the tracking system implicitly uses a form of temporal smoothing by maintaining a history of player features in the deque. This allows the system to match players across frames based on feature continuity.

**Implementation**:
- Features from previous frames are stored and compared with new detections to maintain consistent player IDs
- The use of cosine similarity ensures that small changes in player appearance (e.g., due to lighting or pose) are handled robustly
- The Hungarian algorithm optimizes assignments to minimize abrupt changes in player IDs, effectively acting as a gradient-based optimization over the temporal domain

### 6. Libraries Used
- **Ultralytics YOLO**: For object detection
- **OpenCV**: For image processing and visualization
- **PyTorch**: For model inference and tensor operations
- **NumPy**: For numerical computations
- **SciPy**: For the Hungarian algorithm
- **scikit-learn**: For cosine similarity calculations
- **Matplotlib**: For plotting results
- **Pickle/JSON**: For saving/loading intermediate data
- **Collections (deque)**: For maintaining temporal feature history

## Implementation Options

### Option 1: Cross-Camera Player Mapping

#### Configuration
```python
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

#### Expected Output
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
```

### Option 2: Single Feed Re-Identification

#### Configuration
```python
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

#### Expected Output
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
```

## Alternative Approaches

With more computational power, the following approaches could enhance the system's performance:

### 1. Deep Feature Extraction
- **Approach**: Use a deep learning model (e.g., ResNet, EfficientNet, or a Siamese network) to extract more robust features for re-identification
- **Advantages**: 
  - Deep features capture semantic information (e.g., clothing patterns, body shape) better than color histograms
  - Siamese networks can learn to compare player embeddings directly, improving re-identification accuracy
- **Challenges**: 
  - Requires a large annotated dataset for training
  - Higher computational cost for feature extraction and comparison

### 2. Multi-Object Tracking (MOT) Frameworks
- **Approach**: Adopt established MOT frameworks like DeepSORT or ByteTrack
- **Advantages**: 
  - DeepSORT integrates deep appearance features with Kalman filtering for robust tracking
  - ByteTrack optimizes tracking for high-speed scenarios with occlusion handling
- **Challenges**: 
  - Requires integration of a separate re-identification model
  - Increased memory and compute requirements

### 3. Temporal Smoothing with RNNs or Transformers
- **Approach**: Use Recurrent Neural Networks (RNNs) or Transformers to model temporal dependencies in player trajectories
- **Advantages**: 
  - Captures long-term dependencies in player movement
  - Improves tracking in crowded scenes with occlusions
- **Challenges**: 
  - Requires significant computational resources for training and inference
  - Needs a large dataset with temporal annotations

### 4. Fine-Tuned YOLO Model
- **Approach**: Fine-tune the YOLO model on a sports-specific dataset to improve player detection accuracy
- **Advantages**: 
  - Better detection of players in challenging conditions (e.g., overlapping players, motion blur)
  - Reduces false positives and missed detections
- **Challenges**: 
  - Requires a labeled dataset with player annotations
  - Time-consuming to train and validate

### 5. Optical Flow for Motion Tracking
- **Approach**: Incorporate optical flow to track player motion between frames
- **Advantages**: 
  - Improves tracking by leveraging motion cues
  - Robust to appearance changes caused by lighting or camera angles
- **Challenges**: 
  - Computationally expensive, especially for high-resolution videos
  - May struggle with fast-moving players or low frame rates

### 6. Rolling Gradients with Kalman Filtering
- **Approach**: Implement a Kalman filter to smooth player trajectories and predict future positions, effectively applying rolling gradients to spatial coordinates
- **Advantages**: 
  - Reduces jitter in tracking by modeling player motion as a dynamic system
  - Handles occlusions by predicting positions during temporary losses
- **Challenges**: 
  - Requires tuning of Kalman filter parameters (e.g., process noise, measurement noise)
  - Assumes linear motion, which may not hold for complex player movements

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
```

#### 2. CUDA Memory Issues
```python
# Issue: RuntimeError: CUDA out of memory
# Solutions:
torch.cuda.empty_cache()           # Clear cache
batch_size = 1                     # Reduce batch size
imgsz = 640                        # Reduce input size (default: 1280)
```

#### 3. Video Codec Problems
```bash
# Issue: Cannot read video file
# Solution: Install FFmpeg
# Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg
# Windows: Download from https://ffmpeg.org/
# Mac: brew install ffmpeg
```

## Validation and Testing

### Environment Validation
```python
def validate_environment():
    """Check if environment is properly set up"""
    import sys, torch, cv2, os
    
    checks = {
        'Python': '✓' if sys.version_info >= (3, 11) else '✗ (requires 3.11+)',
        'PyTorch': f"✓ ({torch.__version__})" if torch else '✗',
        'CUDA': f"✓ ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else '○ (CPU only)',
        'OpenCV': f"✓ ({cv2.__version__})" if cv2 else '✗',
        'Model': '✓' if os.path.exists('model.pt') else '✗ (model.pt not found)'
    }
    
    print("Environment Validation:")
    print("=" * 30)
    for component, status in checks.items():
        print(f"{component:10}: {status}")

validate_environment()
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

## Dependencies

### Core Requirements
```txt
Python 3.11+
ultralytics==8.3.160
opencv-python==4.11.0.86
torch==2.6.0+cu124
torchvision>=0.15.0
numpy==2.0.2
scipy==1.15.3
scikit-learn
matplotlib==3.10.0
```

### Optional Enhancements
```txt
gdown>=4.7.0          # Google Drive downloads
tqdm>=4.64.0           # Progress bars
seaborn>=0.11.0        # Enhanced visualizations
pillow>=9.0.0          # Image processing
tensorboard>=2.10.0    # Training visualization
```

### GPU Support
```txt
CUDA 12.4 and cuDNN 9.1 (for GPU support)
```

## Usage

1. Ensure all dependencies are installed and the input video is available
2. Open `liat_ai_Task (1).ipynb` in Jupyter Notebook
3. Execute the cells to:
   - Install the Ultralytics YOLO package
   - Load the YOLO model and process the video
   - Extract features, track players, and visualize results
4. Outputs (e.g., annotated frames or videos) will be saved in the working directory

### Hardware Requirements

#### Minimum Configuration
- **CPU**: Intel i5 / AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional (CPU processing supported)

#### Recommended Configuration  
- **CPU**: Intel i7/i9 / AMD Ryzen 7/9
- **RAM**: 16-32GB for large videos
- **GPU**: NVIDIA GTX 1060 / RTX 3060 or better
- **Storage**: SSD with 10GB+ free space

## Potential Improvements

To enhance the current approach with more computational power:

- **Increase Feature Dimensionality**: Use higher-resolution color histograms or combine with texture features (e.g., Local Binary Patterns)
- **Parallel Processing**: Optimize feature extraction and tracking for parallel execution on GPUs using PyTorch's DataParallel or multiprocessing
- **Occlusion Handling**: Implement occlusion detection by analyzing overlapping bounding boxes and using temporal context to maintain IDs
- **Data Augmentation**: Apply augmentations (e.g., rotation, scaling) during feature extraction to improve robustness to pose and lighting changes
- **Ensemble Features**: Combine multiple feature types (e.g., color histograms, deep embeddings, and motion vectors) for more accurate re-identification
- **Real-Time Processing**: Optimize the pipeline for real-time performance using TensorRT or ONNX for model inference

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

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Good luck with your implementation!** 

For questions or issues, please contact the Liat.ai internship supervisor.
