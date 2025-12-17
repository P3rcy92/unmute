"""
Simple API wrapper for ASL video inference
Can be used as a standalone function or imported into other applications

Optimizations:
- Thread-safe singleton pattern for predictor
- Environment variable configuration
- Lazy initialization with double-checked locking
"""

import base64
import json
import os
import threading
from video_inference import ASLVideoPredictor


# Configuration from environment variables with fallbacks
MODEL_PATH = os.environ.get(
    "ASL_MODEL_PATH",
    '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt'
)
CLASS_LIST_PATH = os.environ.get(
    "ASL_CLASS_LIST_PATH",
    '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt'
)
NUM_CLASSES = int(os.environ.get("ASL_NUM_CLASSES", "2000"))
DEVICE = os.environ.get("ASL_DEVICE", "cuda")

# Thread-safe singleton predictor
_predictor = None
_predictor_lock = threading.Lock()


def get_predictor():
    """
    Get or create the global predictor instance.
    Thread-safe with double-checked locking pattern.
    """
    global _predictor
    
    # Fast path - already initialized
    if _predictor is not None:
        return _predictor
    
    # Slow path - need to initialize
    with _predictor_lock:
        # Double-check after acquiring lock
        if _predictor is None:
            _predictor = ASLVideoPredictor(
                model_path=MODEL_PATH,
                class_list_path=CLASS_LIST_PATH,
                num_classes=NUM_CLASSES,
                device=DEVICE
            )
    return _predictor


def predict_asl_from_base64(base64_video_string, top_k=5):
    """
    Predict ASL sign from base64-encoded video
    
    Args:
        base64_video_string (str): Base64 encoded video
        top_k (int): Number of top predictions to return
        
    Returns:
        dict: Prediction results with format:
            {
                'top_prediction': str,
                'confidence': float,
                'top_k_predictions': [
                    {
                        'rank': int,
                        'label': str,
                        'confidence': float
                    },
                    ...
                ]
            }
    """
    predictor = get_predictor()
    return predictor.predict_from_base64(base64_video_string, top_k=top_k)


def predict_asl_from_file(video_file_path, top_k=5):
    """
    Predict ASL sign from video file
    
    Args:
        video_file_path (str): Path to video file
        top_k (int): Number of top predictions to return
        
    Returns:
        dict: Prediction results (same format as predict_asl_from_base64)
    """
    predictor = get_predictor()
    return predictor.predict_from_file(video_file_path, top_k=top_k)


def video_file_to_base64(video_path):
    """
    Convert video file to base64 string
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        str: Base64 encoded video
    """
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    return base64.b64encode(video_bytes).decode('utf-8')


# Example usage
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_inference_api.py <video_file_path>")
        print("\nThis script demonstrates the simple API for ASL recognition.")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print("Loading model...")
    
    # Method 1: Direct file prediction
    print(f"\nMethod 1: Predicting from file '{video_path}'...")
    result = predict_asl_from_file(video_path, top_k=5)
    
    print(f"\nTop Prediction: {result['top_prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop 5 Predictions:")
    for pred in result['top_k_predictions']:
        print(f"  {pred['rank']}. {pred['label']}: {pred['confidence']:.4f}")
    
    # Method 2: Base64 prediction (commented out to avoid redundant processing)
    # print(f"\n\nMethod 2: Converting to base64 and predicting...")
    # base64_video = video_file_to_base64(video_path)
    # result2 = predict_asl_from_base64(base64_video, top_k=5)
    # print(f"Result: {result2['top_prediction']} ({result2['confidence']:.4f})")
