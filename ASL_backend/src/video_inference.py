"""
Video Inference Script for ASL Recognition
Receives base64-encoded video, preprocesses it, and returns ASL sign prediction
"""

import os
import sys
import base64
import tempfile
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Add WLASL I3D code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../WLASL/code/I3D'))
from pytorch_i3d import InceptionI3d
import videotransforms


class ASLVideoPredictor:
    """ASL Video Recognition Model"""
    
    def __init__(self, model_path, class_list_path, num_classes=2000, device='cuda'):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model checkpoint
            class_list_path: Path to the class list file (label mapping)
            num_classes: Number of classes (default 2000)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.model_path = model_path
        
        # Load class names
        self.class_names = self._load_class_names(class_list_path)
        
        # Initialize model
        self.model = self._load_model()
        
        # Define preprocessing transforms
        self.transforms = transforms.Compose([
            videotransforms.CenterCrop(224)
        ])
        
    def _load_class_names(self, class_list_path):
        """Load class names from file"""
        class_names = {}
        with open(class_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    idx, name = parts
                    class_names[int(idx)] = name
        return class_names
    
    def _load_model(self):
        """Load the I3D model"""
        # Initialize model with ImageNet pretrained weights structure
        model = InceptionI3d(400, in_channels=3)
        
        # Replace logits layer for ASL classes
        model.replace_logits(self.num_classes)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle DataParallel wrapped models
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DataParallel)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        
        # Move to device
        if self.device == 'cuda':
            model = model.cuda()
            model = nn.DataParallel(model)
        
        model.eval()
        return model
    
    def _decode_base64_video(self, base64_string, output_path=None):
        """
        Decode base64 string to video file
        
        Args:
            base64_string: Base64 encoded video
            output_path: Optional path to save video file
            
        Returns:
            Path to the decoded video file
        """
        # Decode base64
        video_data = base64.b64decode(base64_string)
        
        # Create temporary file if no output path specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_path = temp_file.name
            temp_file.close()
        
        # Write video data
        with open(output_path, 'wb') as f:
            f.write(video_data)
        
        return output_path
    
    def _load_video_frames_from_bytes(self, video_bytes: bytes):
        """
        Load and preprocess video frames directly from bytes (in-memory).
        This avoids disk I/O for better performance.
        
        Args:
            video_bytes: Raw video bytes
            
        Returns:
            Preprocessed video tensor (1, 3, T, H, W)
        """
        # Create a temporary file-like object in memory isn't directly supported by cv2
        # So we use a temp file but with a faster approach using memory mapping
        # For now, we'll use a smarter temp file approach with RAM disk if available
        
        # Check if /dev/shm exists (Linux RAM disk)
        import platform
        if platform.system() == 'Linux' and os.path.exists('/dev/shm'):
            temp_dir = '/dev/shm'
        else:
            temp_dir = None
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir)
        temp_path = temp_file.name
        
        try:
            temp_file.write(video_bytes)
            temp_file.close()
            
            return self._load_video_frames(temp_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def _load_video_frames(self, video_path):
        """
        Load and preprocess video frames from file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed video tensor (1, 3, T, H, W)
        """
        vidcap = cv2.VideoCapture(video_path)
        
        if not vidcap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        
        # Get total number of frames
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Read all frames
        for _ in range(total_frames):
            success, img = vidcap.read()
            
            if not success:
                break
            
            # Resize maintaining aspect ratio to have shorter side = 224
            h, w, c = img.shape
            if h < w:
                new_h = 224
                new_w = int(w * (224 / h))
            else:
                new_w = 224
                new_h = int(h * (224 / w))
            
            img = cv2.resize(img, (new_w, new_h))
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Normalize to [-1, 1]
            img = (img / 255.0) * 2 - 1
            
            frames.append(img)
        
        vidcap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames could be read from video")
        
        # Convert to numpy array (T, H, W, C)
        frames = np.array(frames, dtype=np.float32)
        
        # Apply center crop
        frames = self.transforms(frames)
        
        # Convert to tensor and rearrange to (1, C, T, H, W)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        
        return frames
    
    def predict_from_base64(self, base64_video, top_k=5, cleanup=True):
        """
        Predict ASL sign from base64-encoded video
        
        Args:
            base64_video: Base64 encoded video string
            top_k: Number of top predictions to return
            cleanup: Whether to delete temporary video file (deprecated, always cleans up)
            
        Returns:
            Dictionary with predictions
        """
        # Decode base64 directly to bytes
        video_bytes = base64.b64decode(base64_video)
        
        # Load frames from bytes (uses RAM disk if available)
        video_tensor = self._load_video_frames_from_bytes(video_bytes)
        
        # Run inference
        return self.predict_from_tensor(video_tensor, top_k=top_k)
    
    def predict_from_base64_legacy(self, base64_video, top_k=5, cleanup=True):
        """
        Legacy method - Predict ASL sign from base64-encoded video using temp file
        Kept for compatibility.
        """
        video_path = None
        
        try:
            # Decode base64 to video file
            video_path = self._decode_base64_video(base64_video)
            
            # Load and preprocess video
            video_tensor = self._load_video_frames(video_path)
            
            # Run inference
            result = self.predict_from_tensor(video_tensor, top_k=top_k)
            
            return result
            
        finally:
            # Cleanup temporary file
            if cleanup and video_path and os.path.exists(video_path):
                os.remove(video_path)
    
    def predict_from_file(self, video_path, top_k=5):
        """
        Predict ASL sign from video file
        
        Args:
            video_path: Path to video file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess video
        video_tensor = self._load_video_frames(video_path)
        
        # Run inference
        return self.predict_from_tensor(video_tensor, top_k=top_k)
    
    def predict_from_tensor(self, video_tensor, top_k=5):
        """
        Run inference on preprocessed video tensor
        
        Args:
            video_tensor: Preprocessed video tensor (1, C, T, H, W)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        with torch.no_grad():
            # Move to device
            if self.device == 'cuda':
                video_tensor = video_tensor.cuda()
            
            # Forward pass
            per_frame_logits = self.model(video_tensor)
            
            # Average predictions across frames
            predictions = torch.mean(per_frame_logits, dim=2)[0]
            
            # Get top-k predictions
            probs = torch.softmax(predictions, dim=0)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes))
            
            # Convert to numpy
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            # Build results
            results = {
                'top_prediction': self.class_names.get(int(top_indices[0]), 'unknown'),
                'confidence': float(top_probs[0]),
                'top_k_predictions': [
                    {
                        'rank': i + 1,
                        'label': self.class_names.get(int(idx), 'unknown'),
                        'confidence': float(prob)
                    }
                    for i, (idx, prob) in enumerate(zip(top_indices, top_probs))
                ]
            }
            
            return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASL Video Recognition Inference')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--base64', type=str, help='Base64 encoded video string')
    parser.add_argument('--base64_file', type=str, help='Path to file containing base64 encoded video')
    parser.add_argument('--model', type=str, 
                        default='/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--class_list', type=str,
                        default='/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt',
                        help='Path to class list file')
    parser.add_argument('--num_classes', type=int, default=2000, help='Number of classes')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print(f"Loading model from {args.model}...")
    predictor = ASLVideoPredictor(
        model_path=args.model,
        class_list_path=args.class_list,
        num_classes=args.num_classes,
        device=args.device
    )
    print("Model loaded successfully!")
    
    # Run prediction
    if args.video:
        print(f"Processing video file: {args.video}")
        result = predictor.predict_from_file(args.video, top_k=args.top_k)
    elif args.base64:
        print("Processing base64 encoded video...")
        result = predictor.predict_from_base64(args.base64, top_k=args.top_k)
    elif args.base64_file:
        print(f"Reading base64 from file: {args.base64_file}")
        with open(args.base64_file, 'r') as f:
            base64_string = f.read().strip()
        result = predictor.predict_from_base64(base64_string, top_k=args.top_k)
    else:
        print("Error: Please provide either --video, --base64, or --base64_file")
        return
    
    # Print results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Top Prediction: {result['top_prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nTop-K Predictions:")
    for pred in result['top_k_predictions']:
        print(f"  {pred['rank']}. {pred['label']}: {pred['confidence']:.4f}")
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
