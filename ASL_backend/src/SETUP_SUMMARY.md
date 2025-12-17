# ASL Video Inference System - Complete Setup

## 📁 Created Files

The following inference scripts have been created in the `src/` directory:

### Core Scripts
1. **`video_inference.py`** - Main inference module with `ASLVideoPredictor` class
2. **`simple_inference_api.py`** - Simplified Python API for easy integration
3. **`flask_api.py`** - REST API server using Flask
4. **`quick_predict.py`** - Quick command-line prediction script

### Documentation & Examples
5. **`INFERENCE_README.md`** - Complete documentation
6. **`example_usage.py`** - 8 different usage examples
7. **`test_setup.py`** - Setup verification script
8. **`requirements_inference.txt`** - Python dependencies

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /home/akpale/hackatons/little_project_Alex/src
pip install -r requirements_inference.txt
```

### 2. Verify Setup
```bash
python test_setup.py
```

### 3. Run Prediction
```bash
# Simplest method
python quick_predict.py /path/to/video.mp4

# Full control
python video_inference.py --video /path/to/video.mp4 --top_k 5
```

---

## 📊 Model Information

- **Model Path**: `/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt`
- **Architecture**: I3D (Inflated 3D ConvNet)
- **Classes**: 2000 ASL signs
- **Class List**: `/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt`

---

## 💻 Usage Methods

### Method 1: Quick Command-Line
```bash
python quick_predict.py video.mp4
```

### Method 2: Python API
```python
from simple_inference_api import predict_asl_from_base64

# Your base64 video string
base64_video = "your_base64_encoded_video_string"

# Get prediction
result = predict_asl_from_base64(base64_video, top_k=5)

print(f"Sign: {result['top_prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Method 3: REST API Server
```bash
# Start server
python flask_api.py

# Make request (in another terminal)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"video_base64": "YOUR_BASE64_STRING", "top_k": 5}'
```

---

## 🔄 Data Flow

```
Base64 Video String
    ↓
[Decode to temp file]
    ↓
[Extract frames with OpenCV]
    ↓
[Preprocess: resize, crop, normalize]
    ↓
[Convert to tensor (1, 3, T, 224, 224)]
    ↓
[I3D Model Inference]
    ↓
[Average predictions across frames]
    ↓
[Get top-K predictions]
    ↓
Return: {
  "top_prediction": "hello",
  "confidence": 0.85,
  "top_k_predictions": [...]
}
```

---

## 📋 Preprocessing Pipeline

The script automatically handles:

1. **Base64 Decoding** - Converts base64 string to video file
2. **Frame Extraction** - Reads all frames using OpenCV
3. **Resizing** - Scales video so shorter side = 224px (maintains aspect ratio)
4. **Center Cropping** - Crops to 224x224 from center
5. **Color Conversion** - BGR → RGB
6. **Normalization** - Scales pixel values from [0, 255] to [-1, 1]
7. **Tensor Conversion** - Converts to PyTorch tensor (1, C, T, H, W)

---

## 🎯 API Response Format

All prediction functions return:

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
    }
  ]
}
```

---

## 🌐 REST API Endpoints

### `GET /health`
Check API health status

### `POST /predict`
Predict from base64 or file path
```json
{
  "video_base64": "base64_string",  // OR
  "video_path": "/path/to/video.mp4",
  "top_k": 5
}
```

### `POST /predict/file`
Upload video file directly
- Form data: `file` (video file), `top_k` (optional)

### `POST /batch_predict`
Process multiple videos
```json
{
  "videos": [
    {"id": "v1", "video_path": "..."},
    {"id": "v2", "video_base64": "..."}
  ],
  "top_k": 5
}
```

---

## 🔧 Configuration

To use different model or paths, edit these variables:

**In `simple_inference_api.py` - `get_predictor()` function:**
```python
model_path = '/path/to/your/model.pt'
class_list_path = '/path/to/class_list.txt'
num_classes = 2000
device = 'cuda'  # or 'cpu'
```

---

## ⚡ Performance Tips

1. **First Prediction**: Takes 5-10 seconds (model loading)
2. **Subsequent Predictions**: < 1 second per video
3. **GPU vs CPU**: GPU is 10-20x faster
4. **Video Length**: Longer videos take more time
5. **Batch Processing**: Use batch API for multiple videos

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
python video_inference.py --video video.mp4 --device cpu
```

### Module Import Errors
```bash
# Make sure you're in the src directory
cd /home/akpale/hackatons/little_project_Alex/src

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Video Loading Errors
- Verify video format (MP4, AVI, MOV supported)
- Check video is not corrupted
- Ensure base64 encoding is correct

---

## 📦 Integration Examples

### Web Application
```python
from flask import Flask, request, jsonify
from simple_inference_api import predict_asl_from_base64

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    result = predict_asl_from_base64(data['video_base64'])
    return jsonify(result)
```

### Python Script
```python
import base64
from simple_inference_api import predict_asl_from_file

# Predict from file
result = predict_asl_from_file('sign.mp4')
print(f"Prediction: {result['top_prediction']}")

# Or from base64
with open('video.mp4', 'rb') as f:
    video_b64 = base64.b64encode(f.read()).decode()
result = predict_asl_from_base64(video_b64)
```

---

## 📚 Additional Resources

- **Full Documentation**: `INFERENCE_README.md`
- **Usage Examples**: `example_usage.py` (8 examples)
- **Setup Test**: `test_setup.py`
- **Original I3D Code**: `/WLASL/code/I3D/`

---

## ✅ Testing Checklist

- [ ] Dependencies installed (`pip install -r requirements_inference.txt`)
- [ ] Model checkpoint exists
- [ ] Class list file exists
- [ ] CUDA available (or using CPU mode)
- [ ] Setup test passes (`python test_setup.py`)
- [ ] Quick predict works (`python quick_predict.py video.mp4`)

---

## 📞 Support

For issues or questions:
1. Check `INFERENCE_README.md` for detailed documentation
2. Run `python test_setup.py` to diagnose issues
3. Review `example_usage.py` for usage patterns
4. Check that all file paths are correct

---

**Status**: ✅ Ready to use!

**Last Updated**: December 17, 2025
