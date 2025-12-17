"""
Example usage scripts for ASL video inference
"""

import base64
import json


def example_1_predict_from_file():
    """Example 1: Predict from a video file"""
    print("\n" + "="*60)
    print("Example 1: Predict from video file")
    print("="*60)
    
    from simple_inference_api import predict_asl_from_file
    
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    
    # Get prediction
    result = predict_asl_from_file(video_path, top_k=5)
    
    # Print results
    print(f"\nTop Prediction: {result['top_prediction']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    print("\nTop 5 Predictions:")
    for pred in result['top_k_predictions']:
        print(f"  {pred['rank']}. {pred['label']:<20} {pred['confidence']:.4f}")


def example_2_predict_from_base64():
    """Example 2: Predict from base64-encoded video"""
    print("\n" + "="*60)
    print("Example 2: Predict from base64-encoded video")
    print("="*60)
    
    from simple_inference_api import predict_asl_from_base64, video_file_to_base64
    
    video_path = "path/to/your/video.mp4"  # Replace with actual path
    
    # Convert video to base64
    print("Converting video to base64...")
    base64_video = video_file_to_base64(video_path)
    print(f"Base64 length: {len(base64_video)} characters")
    
    # Get prediction
    print("Running prediction...")
    result = predict_asl_from_base64(base64_video, top_k=5)
    
    # Print results
    print(f"\nPredicted Sign: {result['top_prediction']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")


def example_3_use_predictor_class():
    """Example 3: Using the ASLVideoPredictor class directly"""
    print("\n" + "="*60)
    print("Example 3: Using ASLVideoPredictor class")
    print("="*60)
    
    from video_inference import ASLVideoPredictor
    
    # Initialize predictor
    model_path = '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt'
    class_list_path = '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt'
    
    print("Loading model...")
    predictor = ASLVideoPredictor(
        model_path=model_path,
        class_list_path=class_list_path,
        num_classes=2000,
        device='cuda'  # or 'cpu'
    )
    print("Model loaded!")
    
    # Predict from multiple videos
    videos = [
        "video1.mp4",
        "video2.mp4",
        "video3.mp4"
    ]
    
    for video_path in videos:
        print(f"\nProcessing {video_path}...")
        result = predictor.predict_from_file(video_path, top_k=3)
        print(f"  Prediction: {result['top_prediction']} ({result['confidence']:.2%})")


def example_4_rest_api_client():
    """Example 4: Using the REST API"""
    print("\n" + "="*60)
    print("Example 4: REST API Client")
    print("="*60)
    
    import requests
    
    api_url = "http://localhost:5000"
    
    # Health check
    print("Checking API health...")
    response = requests.get(f"{api_url}/health")
    print(f"Status: {response.json()}")
    
    # Predict from file path
    print("\nPredicting from file path...")
    data = {
        "video_path": "/path/to/video.mp4",
        "top_k": 5
    }
    response = requests.post(f"{api_url}/predict", json=data)
    result = response.json()
    
    if result['success']:
        print(f"Prediction: {result['top_prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
    else:
        print(f"Error: {result['error']}")
    
    # Upload file
    print("\nUploading file...")
    with open('/path/to/video.mp4', 'rb') as f:
        files = {'file': f}
        data = {'top_k': 5}
        response = requests.post(f"{api_url}/predict/file", files=files, data=data)
        result = response.json()
    
    if result['success']:
        print(f"Prediction: {result['top_prediction']}")


def example_5_batch_processing():
    """Example 5: Batch processing multiple videos"""
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)
    
    from simple_inference_api import predict_asl_from_file
    import os
    
    # Process all videos in a directory
    video_dir = "/path/to/video/directory"
    results = {}
    
    for filename in os.listdir(video_dir):
        if filename.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, filename)
            print(f"Processing {filename}...")
            
            try:
                result = predict_asl_from_file(video_path, top_k=1)
                results[filename] = {
                    'prediction': result['top_prediction'],
                    'confidence': result['confidence']
                }
                print(f"  -> {result['top_prediction']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"  -> Error: {e}")
                results[filename] = {'error': str(e)}
    
    # Save results
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to batch_results.json")


def example_6_rest_api_batch():
    """Example 6: Batch prediction via REST API"""
    print("\n" + "="*60)
    print("Example 6: REST API Batch Prediction")
    print("="*60)
    
    import requests
    
    api_url = "http://localhost:5000/batch_predict"
    
    # Prepare batch request
    data = {
        "videos": [
            {"id": "video1", "video_path": "/path/to/video1.mp4"},
            {"id": "video2", "video_path": "/path/to/video2.mp4"},
            {"id": "video3", "video_path": "/path/to/video3.mp4"},
        ],
        "top_k": 3
    }
    
    print("Sending batch request...")
    response = requests.post(api_url, json=data)
    result = response.json()
    
    if result['success']:
        print(f"\nProcessed {result['successful']}/{result['total']} videos")
        print("\nResults:")
        for item in result['results']:
            pred = item['prediction']
            print(f"  {item['id']}: {pred['top_prediction']} ({pred['confidence']:.2%})")
        
        if result['errors']:
            print("\nErrors:")
            for error in result['errors']:
                print(f"  {error['id']}: {error['error']}")


def example_7_stream_processing():
    """Example 7: Process video stream or webcam"""
    print("\n" + "="*60)
    print("Example 7: Stream Processing (Conceptual)")
    print("="*60)
    
    print("""
    For real-time processing:
    
    1. Capture video frames from webcam or stream
    2. Buffer frames (e.g., 64 frames)
    3. Preprocess frames to tensor
    4. Run inference using predictor.predict_from_tensor()
    5. Display results
    
    Note: The current model processes entire videos.
    For real-time, you'd need to:
    - Implement sliding window approach
    - Use smaller frame counts
    - Optimize preprocessing
    """)


def example_8_custom_preprocessing():
    """Example 8: Custom preprocessing pipeline"""
    print("\n" + "="*60)
    print("Example 8: Custom Preprocessing")
    print("="*60)
    
    from video_inference import ASLVideoPredictor
    import torch
    import numpy as np
    import cv2
    
    # Load model
    model_path = '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt'
    class_list_path = '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt'
    
    predictor = ASLVideoPredictor(
        model_path=model_path,
        class_list_path=class_list_path,
        num_classes=2000,
        device='cuda'
    )
    
    # Custom preprocessing
    video_path = "path/to/video.mp4"
    
    # 1. Load video with custom settings
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Custom processing: e.g., apply filters, adjust brightness, etc.
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
        
        frames.append(frame)
    
    cap.release()
    
    # 2. Manually preprocess to tensor
    # ... (implement your custom preprocessing)
    
    # 3. Run inference on tensor
    # result = predictor.predict_from_tensor(video_tensor, top_k=5)
    
    print("See code for custom preprocessing implementation")


if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Predict from file', example_1_predict_from_file),
        '2': ('Predict from base64', example_2_predict_from_base64),
        '3': ('Use predictor class', example_3_use_predictor_class),
        '4': ('REST API client', example_4_rest_api_client),
        '5': ('Batch processing', example_5_batch_processing),
        '6': ('REST API batch', example_6_rest_api_batch),
        '7': ('Stream processing', example_7_stream_processing),
        '8': ('Custom preprocessing', example_8_custom_preprocessing),
    }
    
    print("\n" + "="*60)
    print("ASL Video Inference - Example Usage")
    print("="*60)
    print("\nAvailable examples:")
    for key, (description, _) in examples.items():
        print(f"  {key}. {description}")
    print("\nUsage: python example_usage.py <example_number>")
    print("Example: python example_usage.py 1")
    print("\nNote: Update file paths in the code before running!")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            description, func = examples[choice]
            func()
        else:
            print(f"\nInvalid example number: {choice}")
    else:
        print("\nProvide an example number to run (1-8)")
