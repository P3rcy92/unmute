#!/bin/bash
# Quick Reference Commands for ASL Video Inference

echo "============================================================"
echo "ASL Video Inference - Quick Reference"
echo "============================================================"
echo ""

echo "📦 Installation:"
echo "  pip install -r requirements_inference.txt"
echo ""

echo "🧪 Test Setup:"
echo "  python test_setup.py"
echo ""

echo "🚀 Quick Prediction:"
echo "  python quick_predict.py video.mp4"
echo ""

echo "📹 Predict from Video File:"
echo "  python video_inference.py --video video.mp4"
echo "  python video_inference.py --video video.mp4 --top_k 10 --output result.json"
echo ""

echo "🔤 Predict from Base64:"
echo "  python video_inference.py --base64 'BASE64_STRING'"
echo "  python video_inference.py --base64_file video_base64.txt"
echo ""

echo "🌐 Start REST API Server:"
echo "  python flask_api.py"
echo ""

echo "📡 API Requests:"
echo "  # Health check"
echo "  curl http://localhost:5000/health"
echo ""
echo "  # Predict with base64"
echo "  curl -X POST http://localhost:5000/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"video_base64\": \"BASE64_STRING\", \"top_k\": 5}'"
echo ""
echo "  # Upload file"
echo "  curl -X POST http://localhost:5000/predict/file \\"
echo "    -F 'file=@video.mp4' -F 'top_k=5'"
echo ""

echo "🐍 Python API:"
echo "  from simple_inference_api import predict_asl_from_base64"
echo "  result = predict_asl_from_base64(base64_video, top_k=5)"
echo "  print(result['top_prediction'])"
echo ""

echo "📚 Documentation:"
echo "  SETUP_SUMMARY.md      - Complete setup guide"
echo "  INFERENCE_README.md   - Detailed documentation"
echo "  example_usage.py      - 8 usage examples"
echo ""

echo "============================================================"
