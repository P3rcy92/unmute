#!/usr/bin/env python3
"""
Quick Start Script - Simplest way to predict ASL from video
Usage: python quick_predict.py <video_file_path>
"""

import sys
import os

# Check if video path is provided
if len(sys.argv) < 2:
    print("Usage: python quick_predict.py <video_file_path>")
    print("\nExample: python quick_predict.py /path/to/sign_video.mp4")
    sys.exit(1)

video_path = sys.argv[1]

# Check if file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found: {video_path}")
    sys.exit(1)

print("="*60)
print("ASL Video Recognition - Quick Predict")
print("="*60)
print(f"\nVideo: {video_path}")
print("Loading model...")

# Import the prediction function
try:
    from simple_inference_api import predict_asl_from_file
except ImportError as e:
    print(f"Error importing inference API: {e}")
    print("\nMake sure you're in the correct directory and dependencies are installed.")
    print("Run: pip install -r requirements_inference.txt")
    sys.exit(1)

# Run prediction
try:
    print("Running prediction...\n")
    result = predict_asl_from_file(video_path, top_k=5)
    
    # Display results
    print("="*60)
    print("RESULT")
    print("="*60)
    print(f"\n🎯 Predicted Sign: {result['top_prediction'].upper()}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    
    print("\n📋 Top 5 Predictions:")
    print("-" * 60)
    for pred in result['top_k_predictions']:
        bar_length = int(pred['confidence'] * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{pred['rank']}. {pred['label']:<20} {bar} {pred['confidence']:.2%}")
    
    print("\n" + "="*60)
    print("✓ Prediction complete!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error during prediction: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
