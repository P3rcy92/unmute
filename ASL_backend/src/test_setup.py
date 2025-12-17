"""
Test script to verify the inference setup
"""

import os
import sys


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV (cv2)',
        'numpy': 'NumPy',
        'flask': 'Flask'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch torchvision opencv-python numpy flask")
        return False
    
    print("All dependencies found!")
    return True


def check_paths():
    """Check if model and class list files exist"""
    print("\nChecking file paths...")
    
    model_path = '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt'
    class_list_path = '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt'
    i3d_code_path = '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D'
    
    paths = {
        'Model checkpoint': model_path,
        'Class list': class_list_path,
        'I3D code directory': i3d_code_path
    }
    
    all_exist = True
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT FOUND - {path}")
            all_exist = False
    
    if not all_exist:
        print("\nSome required files are missing!")
        return False
    
    print("All required files found!")
    return True


def check_cuda():
    """Check CUDA availability"""
    print("\nChecking CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print(f"  ! CUDA not available - will use CPU (slower)")
            print(f"  Consider using --device cpu flag")
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")


def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        sys.path.insert(0, '/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D')
        from pytorch_i3d import InceptionI3d
        
        print("  ✓ I3D module imported successfully")
        
        # Try creating model instance
        model = InceptionI3d(400, in_channels=3)
        print("  ✓ Model instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False


def test_inference_api():
    """Test if the inference API can be imported"""
    print("\nTesting inference API...")
    
    try:
        from simple_inference_api import get_predictor
        print("  ✓ Inference API imported successfully")
        
        # Try loading the predictor (this will load the full model)
        print("  Loading full model (this may take a moment)...")
        predictor = get_predictor()
        print("  ✓ Predictor loaded successfully!")
        print(f"  Model has {predictor.num_classes} classes")
        print(f"  Device: {predictor.device}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("ASL Video Inference - Setup Verification")
    print("="*60)
    
    tests = [
        check_dependencies,
        check_paths,
        check_cuda,
        test_model_loading,
        test_inference_api
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            results.append(False)
        print()
    
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    if all(results):
        print("✓ All tests passed! System is ready for inference.")
        print("\nYou can now:")
        print("  1. Run predictions: python video_inference.py --video path/to/video.mp4")
        print("  2. Start API server: python flask_api.py")
        print("  3. Use Python API: from simple_inference_api import predict_asl_from_file")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install torch torchvision opencv-python numpy flask")
        print("  - Check file paths are correct")
        print("  - Ensure model checkpoint exists")
    
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
