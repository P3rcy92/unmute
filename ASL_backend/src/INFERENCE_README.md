# ASL Video Inference Scripts

This directory contains scripts for running ASL (American Sign Language) recognition inference on video files using the trained I3D model.

## Files

- **`video_inference.py`** - Main inference module with the `ASLVideoPredictor` class
- **`simple_inference_api.py`** - Simple Python API for easy integration
- **`flask_api.py`** - REST API server using Flask
- **`example_usage.py`** - Example scripts showing different usage patterns

## Setup

### Prerequisites

```bash
pip install torch torchvision opencv-python numpy flask
```

### Model and Data Paths

The scripts are configured to use:
- **Model**: `/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt`
- **Class List**: `/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt`

## Usage

### 1. Command-Line Interface

#### Predict from video file:
```bash
python video_inference.py --video /path/to/video.mp4
```

#### Predict from base64 string:
```bash
python video_inference.py --base64 "base64_encoded_video_string"
```

#### Predict from base64 file:
```bash
python video_inference.py --base64_file video_base64.txt
```

#### Additional options:
```bash
python video_inference.py \
    --video /path/to/video.mp4 \
    --top_k 10 \
    --device cuda \
    --output results.json
```

### 2. Python API

```python
from simple_inference_api import predict_asl_from_base64, predict_asl_from_file

# Method 1: From file
result = predict_asl_from_file('/path/to/video.mp4', top_k=5)
print(f"Prediction: {result['top_prediction']}")
print(f"Confidence: {result['confidence']:.4f}")

# Method 2: From base64 string
base64_video = "your_base64_encoded_video"
result = predict_asl_from_base64(base64_video, top_k=5)
print(f"Prediction: {result['top_prediction']}")
```

### 3. REST API Server

#### Start the server:
```bash
python flask_api.py
```

The server runs on `http://localhost:5000`

#### API Endpoints:

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Predict with Base64:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "video_base64": "base64_encoded_video",
    "top_k": 5
  }'
```

**Predict with File Path:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "top_k": 5
  }'
```

**Upload Video File:**
```bash
curl -X POST http://localhost:5000/predict/file \
  -F "file=@/path/to/video.mp4" \
  -F "top_k=5"
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "videos": [
      {"id": "video1", "video_path": "/path/to/video1.mp4"},
      {"id": "video2", "video_base64": "base64_string"}
    ],
    "top_k": 5
  }'
```

## Response Format

All prediction functions return a dictionary with the following structure:

```json
{
  "top_prediction": "hello",
  "confidence": 0.8523,
  "top_k_predictions": [
    {
      "rank": 1,
      "label": "hello",
      "confidence": 0.8523
    },
    {
      "rank": 2,
      "label": "hi",
      "confidence": 0.0892
    },
    ...
  ]
}
```

## Video Preprocessing

The scripts automatically handle:
- Video decoding from base64
- Frame extraction
- Resizing (shorter side to 224px)
- Center cropping (224x224)
- Normalization to [-1, 1]
- Tensor conversion and batching

## Model Details

- **Architecture**: I3D (Inflated 3D ConvNet)
- **Classes**: 2000 ASL signs
- **Input**: RGB video frames
- **Preprocessing**: Center crop to 224x224, normalized to [-1, 1]

## Example Integration

```python
import base64
from simple_inference_api import predict_asl_from_base64

# Read video file
with open('sign_video.mp4', 'rb') as f:
    video_bytes = f.read()

# Encode to base64
video_base64 = base64.b64encode(video_bytes).decode('utf-8')

# Get prediction
result = predict_asl_from_base64(video_base64, top_k=5)

# Use the result
print(f"The sign is: {result['top_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Error Handling

All functions include error handling for:
- Invalid video formats
- Corrupt video files
- Empty videos
- Model loading errors
- GPU/CPU device issues

## Performance Notes

- First prediction is slower due to model loading (can take 5-10 seconds)
- Subsequent predictions are faster (< 1 second per video)
- GPU inference is significantly faster than CPU
- Video length affects inference time (longer videos = more processing)

## Troubleshooting

**CUDA out of memory:**
- Use `--device cpu` flag
- Process shorter videos
- Reduce batch size internally

**Module import errors:**
- Ensure WLASL/code/I3D is in the Python path
- Check all dependencies are installed

**Video loading errors:**
- Verify video file is not corrupted
- Check video format is supported by OpenCV
- Ensure base64 string is properly encoded
