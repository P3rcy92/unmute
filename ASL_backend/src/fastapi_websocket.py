"""
FastAPI WebSocket Server for Real-time ASL Sign Language Recognition
Compatible with the unmute frontend WebSocket contract.

Receives: {"type": "frame", "data": "data:image/jpeg;base64,...", "timestamp": 1702834567890}
Sends: {"type": "word", "word": "Hello"}

Optimizations applied:
- Async inference using thread pool executor to prevent blocking event loop
- Adaptive inference triggering with frame skipping when backlogged
- Connection health monitoring with timeouts
- Faster JSON serialization using orjson
- Modern FastAPI lifespan pattern
- Frame queue with backpressure handling
"""

import os
import sys
import base64
import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Try to use orjson for faster JSON serialization, fallback to standard json
try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
    def json_loads(s):
        return orjson.loads(s)
except ImportError:
    import json
    json_dumps = json.dumps
    json_loads = json.loads

# Add WLASL I3D code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../WLASL/code/I3D'))
from pytorch_i3d import InceptionI3d
import videotransforms

# Create formatters
info_formatter = logging.Formatter('%(asctime)s | ✅ %(message)s', datefmt='%H:%M:%S')
error_formatter = logging.Formatter('%(asctime)s | ❌ ERROR | %(message)s', datefmt='%H:%M:%S')
file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Clear existing handlers
logger.handlers = []

# Console handler for INFO and above (good logs)
info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_handler.addFilter(lambda record: record.levelno < logging.WARNING)  # Only INFO level
info_handler.setFormatter(info_formatter)

# Console handler for WARNING and above (error logs)  
error_handler = logging.StreamHandler()
error_handler.setLevel(logging.WARNING)
error_handler.setFormatter(error_formatter)

# File handler for all logs
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)

# Info file handler
info_file_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
info_file_handler.setLevel(logging.INFO)
info_file_handler.addFilter(lambda record: record.levelno < logging.WARNING)
info_file_handler.setFormatter(file_formatter)

# Error file handler
error_file_handler = logging.FileHandler(os.path.join(log_dir, 'error.log'))
error_file_handler.setLevel(logging.WARNING)
error_file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(info_handler)
logger.addHandler(error_handler)
logger.addHandler(info_file_handler)
logger.addHandler(error_file_handler)

# Suppress other loggers' noise
logging.getLogger('pyngrok').setLevel(logging.WARNING)
logging.getLogger('uvicorn').setLevel(logging.INFO)

# Configuration - Use environment variables with fallbacks for easier deployment
MODEL_PATH = os.environ.get(
    "ASL_MODEL_PATH",
    "/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/checkpoints/nslt_2000_003036_0.466481.pt"
)
CLASS_LIST_PATH = os.environ.get(
    "ASL_CLASS_LIST_PATH",
    "/home/akpale/hackatons/little_project_Alex/WLASL/code/I3D/preprocess/wlasl_class_list.txt"
)
NUM_CLASSES = int(os.environ.get("ASL_NUM_CLASSES", "2000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Frame buffer settings
FRAME_BUFFER_SIZE = 48  # Number of frames to accumulate before inference (about 2 seconds at 25 FPS)
MIN_FRAMES_FOR_INFERENCE = 16  # Minimum frames needed for inference
TARGET_FPS = 25  # Expected frames per second from frontend

# Performance settings
INFERENCE_THREAD_POOL_SIZE = 2  # Number of threads for inference (1-2 is usually optimal for GPU)
CONNECTION_TIMEOUT_SECONDS = 300  # 5 minutes timeout for idle connections
MAX_FRAME_QUEUE_SIZE = 100  # Maximum frames to queue before dropping
ADAPTIVE_INFERENCE_COOLDOWN = 0.5  # Minimum seconds between inferences

# Thread pool for running inference without blocking the event loop
inference_executor = ThreadPoolExecutor(max_workers=INFERENCE_THREAD_POOL_SIZE, thread_name_prefix="inference")


class ASLRealtimePredictor:
    """Real-time ASL Recognition using I3D model"""
    
    def __init__(self, model_path: str, class_list_path: str, num_classes: int = 2000, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        
        logger.info(f"Loading model on {self.device}...")
        
        # Load class names
        self.class_names = self._load_class_names(class_list_path)
        logger.info(f"Loaded {len(self.class_names)} class names")
        
        # Load model
        self.model = self._load_model(model_path)
        logger.info("Model loaded successfully!")
        
        # Define preprocessing transforms
        self.center_crop = videotransforms.CenterCrop(224)
        
    def _load_class_names(self, class_list_path: str) -> dict:
        """Load class names from file"""
        class_names = {}
        with open(class_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    idx, name = parts
                    class_names[int(idx)] = name
        return class_names
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the I3D model"""
        from collections import OrderedDict
        
        # Initialize model
        model = InceptionI3d(400, in_channels=3)
        model.replace_logits(self.num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        
        # Move to device
        if self.device == "cuda":
            model = model.cuda()
            model = nn.DataParallel(model)
        
        model.eval()
        return model
    
    def preprocess_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for the model.
        
        Args:
            image: BGR image from cv2 (H, W, C)
            
        Returns:
            Preprocessed frame (H, W, C) normalized to [-1, 1]
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to have shorter side = 256 (gives room for center crop)
        h, w, c = image.shape
        if h < w:
            new_h = 256
            new_w = int(w * (256 / h))
        else:
            new_w = 256
            new_h = int(h * (256 / w))
        
        image = cv2.resize(image, (new_w, new_h))
        
        # Normalize to [-1, 1]
        image = (image / 255.0) * 2 - 1
        
        return image.astype(np.float32)
    
    def predict_from_frames(self, frames: list, top_k: int = 5) -> dict:
        """
        Run inference on a list of preprocessed frames.
        
        Args:
            frames: List of preprocessed frames (each H, W, C)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        if len(frames) < MIN_FRAMES_FOR_INFERENCE:
            return None
        
        # Stack frames: (T, H, W, C)
        frames_array = np.array(frames, dtype=np.float32)
        
        # Apply center crop
        frames_array = self.center_crop(frames_array)
        
        # Convert to tensor: (1, C, T, H, W)
        video_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2).unsqueeze(0)
        
        with torch.no_grad():
            if self.device == "cuda":
                video_tensor = video_tensor.cuda()
            
            # Forward pass
            per_frame_logits = self.model(video_tensor)
            
            # Average predictions across frames
            predictions = torch.mean(per_frame_logits, dim=2)[0]
            
            # Get probabilities
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
    
    def warmup(self):
        """
        Warm up the model with a dummy inference to initialize CUDA kernels.
        This reduces latency on the first real inference.
        """
        logger.info("Warming up model with dummy inference...")
        dummy_frames = [np.zeros((256, 256, 3), dtype=np.float32) for _ in range(MIN_FRAMES_FOR_INFERENCE)]
        try:
            self.predict_from_frames(dummy_frames)
            logger.info("Model warmup complete!")
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")


class ConnectionState:
    """State tracking for a single WebSocket connection"""
    __slots__ = ['frame_buffer', 'last_activity', 'inference_in_progress', 
                 'last_inference_time', 'frames_since_inference', 'lock']
    
    def __init__(self):
        self.frame_buffer: deque = deque(maxlen=FRAME_BUFFER_SIZE)
        self.last_activity: datetime = datetime.now()
        self.inference_in_progress: bool = False
        self.last_inference_time: float = 0.0
        self.frames_since_inference: int = 0
        self.lock: threading.Lock = threading.Lock()


class ConnectionManager:
    """Manage WebSocket connections with improved state tracking"""
    
    def __init__(self):
        self.active_connections: dict[WebSocket, ConnectionState] = {}
        self._lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections[websocket] = ConnectionState()
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                del self.active_connections[websocket]
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    def get_connection_state(self, websocket: WebSocket) -> Optional[ConnectionState]:
        return self.active_connections.get(websocket)
    
    def get_frame_buffer(self, websocket: WebSocket) -> deque:
        state = self.active_connections.get(websocket)
        return state.frame_buffer if state else deque(maxlen=FRAME_BUFFER_SIZE)
    
    def update_activity(self, websocket: WebSocket):
        state = self.active_connections.get(websocket)
        if state:
            state.last_activity = datetime.now()
    
    async def send_word(self, websocket: WebSocket, word: str, confidence: float = None):
        """Send a recognized word to the client with optional confidence"""
        message = {"type": "word", "word": word}
        if confidence is not None:
            message["confidence"] = round(confidence, 4)
        await websocket.send_text(json_dumps(message))
    
    def get_stale_connections(self, timeout_seconds: int = CONNECTION_TIMEOUT_SECONDS) -> list[WebSocket]:
        """Get connections that have been idle for too long"""
        now = datetime.now()
        stale = []
        with self._lock:
            for ws, state in self.active_connections.items():
                if (now - state.last_activity).total_seconds() > timeout_seconds:
                    stale.append(ws)
        return stale


# Initialize connection manager
manager = ConnectionManager()

# Initialize predictor (will be loaded on startup)
predictor: Optional[ASLRealtimePredictor] = None

# Background task for cleanup
cleanup_task: Optional[asyncio.Task] = None


async def cleanup_stale_connections():
    """Periodically clean up stale connections"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            stale = manager.get_stale_connections()
            for ws in stale:
                logger.info(f"Closing stale connection due to inactivity")
                try:
                    await ws.close(code=1000, reason="Connection timeout due to inactivity")
                except Exception:
                    pass
                manager.disconnect(ws)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan handler for startup and shutdown"""
    global predictor, cleanup_task
    
    # Startup
    logger.info("Starting up... Loading ASL model...")
    predictor = ASLRealtimePredictor(
        model_path=MODEL_PATH,
        class_list_path=CLASS_LIST_PATH,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )
    
    # Warm up the model to reduce first-inference latency
    predictor.warmup()
    logger.info("Model loaded and ready!")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_stale_connections())
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down...")
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    
    # Shutdown thread pool
    inference_executor.shutdown(wait=False)
    logger.info("Shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="ASL Sign Language Recognition API",
    description="Real-time sign language recognition using I3D model via WebSocket",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": "I3D ASL 2000",
        "device": DEVICE,
        "message": "ASL Sign Language Recognition API is running",
        "websocket_endpoint": "/ws"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }


def decode_base64_frame(frame_data: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded JPEG frame.
    
    Args:
        frame_data: Base64 string, possibly with data URI prefix
        
    Returns:
        Decoded image as numpy array (BGR), or None if failed
    """
    try:
        # Handle data URI format: "data:image/jpeg;base64,..."
        if "," in frame_data:
            base64_string = frame_data.split(",")[1]
        else:
            base64_string = frame_data
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None


async def run_inference_async(frames_list: list) -> Optional[dict]:
    """
    Run inference in a thread pool to avoid blocking the event loop.
    This is crucial for maintaining WebSocket responsiveness.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            inference_executor,
            predictor.predict_from_frames,
            frames_list
        )
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return None


def should_run_inference(state: ConnectionState, current_time: float) -> bool:
    """
    Determine if we should run inference based on adaptive criteria.
    Returns True if:
    - Enough frames have been collected
    - Not currently running inference
    - Minimum cooldown has passed
    """
    if state.inference_in_progress:
        return False
    
    if len(state.frame_buffer) < MIN_FRAMES_FOR_INFERENCE:
        return False
    
    time_since_last = current_time - state.last_inference_time
    if time_since_last < ADAPTIVE_INFERENCE_COOLDOWN:
        return False
    
    # Run inference every TARGET_FPS frames or if buffer is getting full
    buffer_fullness = len(state.frame_buffer) / FRAME_BUFFER_SIZE
    frames_trigger = state.frames_since_inference >= TARGET_FPS
    buffer_pressure = buffer_fullness > 0.8
    
    return frames_trigger or buffer_pressure


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sign language recognition.
    
    Optimized for reactivity with:
    - Non-blocking async inference
    - Adaptive inference triggering
    - Backpressure handling
    
    Receives:
        {
            "type": "frame",
            "data": "data:image/jpeg;base64,...",
            "timestamp": 1702834567890
        }
    
    Sends:
        {
            "type": "word",
            "word": "Hello",
            "confidence": 0.85
        }
    """
    await manager.connect(websocket)
    
    client_info = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ Client connected: {client_info}")
    logger.info(f"{'='*50}\n")
    
    state = manager.get_connection_state(websocket)
    if not state:
        logger.error("Failed to get connection state")
        return
    
    last_prediction = None
    frame_count = 0
    words_sent = 0
    dropped_frames = 0
    start_time = datetime.now()
    last_log_time = start_time
    
    # Task for concurrent inference
    inference_task: Optional[asyncio.Task] = None
    
    async def process_inference():
        """Process inference and send result - runs concurrently with frame reception"""
        nonlocal words_sent, last_prediction
        
        try:
            state.inference_in_progress = True
            frames_list = list(state.frame_buffer)
            
            logger.info(f"🧠 Running async inference with {len(frames_list)} frames...")
            
            result = await run_inference_async(frames_list)
            
            if result and websocket in manager.active_connections:
                logger.info(f"📊 Prediction: {result['top_prediction']} (confidence: {result['confidence']:.2%})")
                
                # Log top 3 predictions
                for pred in result['top_k_predictions'][:3]:
                    logger.info(f"   {pred['rank']}. {pred['label']}: {pred['confidence']:.2%}")
                
                current_prediction = result['top_prediction']
                confidence = result['confidence']
                
                # Send prediction with confidence
                words_sent += 1
                await manager.send_word(websocket, current_prediction, confidence)
                
                logger.info(f"📤 Sent word: \"{current_prediction}\" (confidence: {confidence:.2%})")
                last_prediction = current_prediction
            elif not result:
                logger.warning("No result from inference")
                
        except Exception as e:
            logger.error(f"Error in inference task: {e}")
        finally:
            state.inference_in_progress = False
            state.last_inference_time = asyncio.get_event_loop().time()
            state.frames_since_inference = 0
    
    try:
        while True:
            # Receive message from client with timeout for connection health
            try:
                # Use asyncio.wait_for for timeout handling
                raw_data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=CONNECTION_TIMEOUT_SECONDS
                )
                data = json_loads(raw_data)
                manager.update_activity(websocket)
                
            except asyncio.TimeoutError:
                logger.info(f"Connection timeout for {client_info}")
                break
            except Exception as e:
                error_str = str(e).lower()
                if "disconnect" in error_str or "closed" in error_str:
                    break
                logger.warning(f"Error receiving message: {e}")
                break
            
            if data.get("type") != "frame":
                # Handle other message types (e.g., ping/pong for keep-alive)
                if data.get("type") == "ping":
                    await websocket.send_text(json_dumps({"type": "pong"}))
                continue
            
            frame_count += 1
            state.frames_since_inference += 1
            frame_data = data.get("data", "")
            
            if not frame_data:
                continue
            
            # Calculate frame size
            frame_size_kb = len(frame_data) / 1024
            
            # Log statistics every second
            current_time = datetime.now()
            if (current_time - last_log_time).total_seconds() >= 1.0:
                elapsed = (current_time - start_time).total_seconds()
                fps = frame_count / elapsed if elapsed > 0 else 0
                buffer_size = len(state.frame_buffer)
                logger.info(
                    f"📥 Frames: {frame_count} | FPS: {fps:.1f} | "
                    f"Buffer: {buffer_size}/{FRAME_BUFFER_SIZE} | "
                    f"Words: {words_sent} | Dropped: {dropped_frames}"
                )
                last_log_time = current_time
            
            # Log first frame details
            if frame_count == 1:
                logger.info(f"📷 First frame received! Data length: {len(frame_data)} chars")
            
            # Decode the frame
            image = decode_base64_frame(frame_data)
            
            if image is None:
                logger.warning(f"Failed to decode frame #{frame_count}")
                continue
            
            # Log image dimensions on first successful decode
            if frame_count == 1:
                logger.info(f"🖼️ Frame decoded successfully: {image.shape}")
            
            # Preprocess and add to buffer with backpressure handling
            preprocessed_frame = predictor.preprocess_frame(image)
            
            # Check for buffer overflow (backpressure)
            if len(state.frame_buffer) >= MAX_FRAME_QUEUE_SIZE and state.inference_in_progress:
                dropped_frames += 1
                # Skip adding this frame to prevent memory issues
                continue
            
            state.frame_buffer.append(preprocessed_frame)
            
            # Check if we should trigger inference (adaptive)
            current_loop_time = asyncio.get_event_loop().time()
            if should_run_inference(state, current_loop_time):
                # Cancel any pending inference task that hasn't started yet
                if inference_task and not inference_task.done():
                    # Let it complete, don't cancel mid-inference
                    pass
                
                # Start new inference task concurrently
                inference_task = asyncio.create_task(process_inference())
    
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
    
    finally:
        # Cancel any pending inference
        if inference_task and not inference_task.done():
            inference_task.cancel()
            try:
                await inference_task
            except asyncio.CancelledError:
                pass
        
        # Always run cleanup
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'='*50}")
        logger.info(f"👋 Client disconnected: {client_info}")
        logger.info(f"📊 Session statistics:")
        logger.info(f"   - Duration: {elapsed:.1f}s")
        logger.info(f"   - Frames received: {frame_count}")
        if elapsed > 0:
            logger.info(f"   - Average FPS: {frame_count/elapsed:.1f}")
        logger.info(f"   - Words sent: {words_sent}")
        logger.info(f"   - Frames dropped: {dropped_frames}")
        logger.info(f"{'='*50}\n")
        manager.disconnect(websocket)


# Alternative WebSocket endpoint on root for compatibility
@app.websocket("/")
async def websocket_root(websocket: WebSocket):
    """WebSocket on root (redirect to /ws)"""
    await websocket_endpoint(websocket)


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ASL Recognition WebSocket Server...")
    logger.info(f"Model checkpoint: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    
    uvicorn.run(
        "fastapi_websocket:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
