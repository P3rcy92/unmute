"""
Flask REST API for ASL Video Recognition
Provides HTTP endpoints for video prediction

Optimizations:
- Thread pool for concurrent batch processing
- Request timeout handling
- Improved error responses
"""

from flask import Flask, request, jsonify
import base64
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import wraps
import time

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))
from simple_inference_api import predict_asl_from_base64, predict_asl_from_file, get_predictor

# Configuration
MAX_WORKERS = 4  # Thread pool size for batch processing
REQUEST_TIMEOUT = 60  # Seconds
BATCH_TIMEOUT = 120  # Seconds for batch requests

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="flask_predict")

app = Flask(__name__)


def timed_request(timeout_seconds=REQUEST_TIMEOUT):
    """Decorator to add timeout to request handlers"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = f(*args, **kwargs)
            elapsed = time.time() - start_time
            # Add timing info to response headers
            if isinstance(result, tuple):
                response, status_code = result
            else:
                response = result
                status_code = 200
            return response, status_code
        return wrapper
    return decorator


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'I3D ASL 2000',
        'version': '1.0'
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict ASL sign from video
    
    Request body (JSON):
        {
            "video_base64": "<base64-encoded-video-string>",
            "top_k": 5  // optional, default 5
        }
    
    OR
    
        {
            "video_path": "/path/to/video.mp4",
            "top_k": 5  // optional, default 5
        }
    
    Response (JSON):
        {
            "success": true,
            "top_prediction": "hello",
            "confidence": 0.85,
            "top_k_predictions": [
                {
                    "rank": 1,
                    "label": "hello",
                    "confidence": 0.85
                },
                ...
            ]
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        top_k = data.get('top_k', 5)
        
        # Check which input method is used
        if 'video_base64' in data:
            video_base64 = data['video_base64']
            
            if not video_base64:
                return jsonify({
                    'success': False,
                    'error': 'video_base64 is empty'
                }), 400
            
            # Run prediction
            result = predict_asl_from_base64(video_base64, top_k=top_k)
            
        elif 'video_path' in data:
            video_path = data['video_path']
            
            if not video_path:
                return jsonify({
                    'success': False,
                    'error': 'video_path is empty'
                }), 400
            
            if not os.path.exists(video_path):
                return jsonify({
                    'success': False,
                    'error': f'Video file not found: {video_path}'
                }), 400
            
            # Run prediction
            result = predict_asl_from_file(video_path, top_k=top_k)
            
        else:
            return jsonify({
                'success': False,
                'error': 'Must provide either video_base64 or video_path'
            }), 400
        
        # Add success flag
        result['success'] = True
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/file', methods=['POST'])
def predict_file():
    """
    Predict ASL sign from uploaded video file
    
    Form data:
        - file: video file (multipart/form-data)
        - top_k: number of top predictions (optional, default 5)
    
    Response: Same as /predict endpoint
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400
        
        top_k = int(request.form.get('top_k', 5))
        
        # Read file content and encode as base64
        video_bytes = file.read()
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        
        # Run prediction
        result = predict_asl_from_base64(video_base64, top_k=top_k)
        result['success'] = True
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def process_single_video(video_data: dict, top_k: int) -> dict:
    """Process a single video - used for concurrent batch processing"""
    video_id = video_data.get('id', 'unknown')
    
    try:
        if 'video_base64' in video_data:
            result = predict_asl_from_base64(video_data['video_base64'], top_k=top_k)
        elif 'video_path' in video_data:
            result = predict_asl_from_file(video_data['video_path'], top_k=top_k)
        else:
            raise ValueError('Must provide either video_base64 or video_path')
        
        return {
            'id': video_id,
            'prediction': result,
            'success': True
        }
        
    except Exception as e:
        return {
            'id': video_id,
            'error': str(e),
            'success': False
        }


@app.route('/batch_predict', methods=['POST'])
@timed_request(timeout_seconds=BATCH_TIMEOUT)
def batch_predict():
    """
    Predict ASL signs from multiple videos CONCURRENTLY
    
    Request body (JSON):
        {
            "videos": [
                {
                    "id": "video1",
                    "video_base64": "<base64-string>"
                },
                {
                    "id": "video2",
                    "video_path": "/path/to/video.mp4"
                }
            ],
            "top_k": 5,  // optional, default 5
            "parallel": true  // optional, default true - use concurrent processing
        }
    
    Response (JSON):
        {
            "success": true,
            "results": [
                {
                    "id": "video1",
                    "prediction": {...}
                },
                ...
            ],
            "errors": [
                {
                    "id": "video2",
                    "error": "error message"
                }
            ],
            "processing_time_seconds": 2.5
        }
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'videos' not in data:
            return jsonify({
                'success': False,
                'error': 'No videos provided'
            }), 400
        
        videos = data['videos']
        top_k = data.get('top_k', 5)
        use_parallel = data.get('parallel', True)
        
        results = []
        errors = []
        
        if use_parallel and len(videos) > 1:
            # Process videos concurrently using thread pool
            futures = []
            for video_data in videos:
                future = executor.submit(process_single_video, video_data, top_k)
                futures.append(future)
            
            # Collect results with timeout
            for future in futures:
                try:
                    result = future.result(timeout=BATCH_TIMEOUT)
                    if result['success']:
                        results.append({
                            'id': result['id'],
                            'prediction': result['prediction']
                        })
                    else:
                        errors.append({
                            'id': result['id'],
                            'error': result['error']
                        })
                except FuturesTimeoutError:
                    errors.append({
                        'id': 'unknown',
                        'error': 'Processing timeout'
                    })
                except Exception as e:
                    errors.append({
                        'id': 'unknown',
                        'error': str(e)
                    })
        else:
            # Sequential processing for single video or when parallel is disabled
            for video_data in videos:
                result = process_single_video(video_data, top_k)
                if result['success']:
                    results.append({
                        'id': result['id'],
                        'prediction': result['prediction']
                    })
                else:
                    errors.append({
                        'id': result['id'],
                        'error': result['error']
                    })
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'results': results,
            'errors': errors,
            'total': len(videos),
            'successful': len(results),
            'failed': len(errors),
            'processing_time_seconds': round(processing_time, 3)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Print startup message
    print("=" * 60)
    print("ASL Video Recognition API (Optimized)")
    print("=" * 60)
    print("\nLoading model...")
    
    # Pre-load and warm up the model
    predictor = get_predictor()
    print("Model loaded successfully!")
    
    # Warm up with a dummy inference
    print("Warming up model...")
    import numpy as np
    try:
        # Create a minimal dummy video tensor for warmup
        dummy_tensor = np.zeros((1, 3, 16, 224, 224), dtype=np.float32)
        import torch
        dummy_tensor = torch.from_numpy(dummy_tensor)
        if torch.cuda.is_available():
            dummy_tensor = dummy_tensor.cuda()
        with torch.no_grad():
            _ = predictor.model(dummy_tensor)
        print("Model warmup complete!")
    except Exception as e:
        print(f"Warmup skipped: {e}")
    
    print("\nAvailable endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /predict             - Predict from JSON (base64 or path)")
    print("  POST /predict/file        - Predict from uploaded file")
    print("  POST /batch_predict       - Batch prediction (concurrent)")
    print(f"\nThread pool size: {MAX_WORKERS}")
    print(f"Request timeout: {REQUEST_TIMEOUT}s")
    print("\nStarting server...")
    print("=" * 60)
    
    # Run server - consider using gunicorn in production
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 flask_api:app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
