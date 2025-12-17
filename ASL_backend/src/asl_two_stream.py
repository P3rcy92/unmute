import os
import cv2
import math
import glob
import random
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -----------------------------
# Config
# -----------------------------
NUM_FRAMES = 32            # frames per clip
FRAME_SIZE = 224           # resize to 224x224
FLOW_CLIP_LEN = NUM_FRAMES - 1  # flow is defined between pairs
BATCH_SIZE = 4
EPOCHS = 5
LR = 1e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Utils: video loading and sampling
# -----------------------------
def read_video_frames(path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    ok = True
    while ok:
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

def uniform_sample_indices(n_total: int, n_samples: int) -> List[int]:
    if n_total <= n_samples:
        # pad by repeating last
        idx = list(range(n_total)) + [n_total-1] * (n_samples - n_total)
        return idx
    # uniform sampling across the full length
    step = n_total / n_samples
    return [int(i * step) for i in range(n_samples)]

def preprocess_rgb_clip(frames: List[np.ndarray], size: int = FRAME_SIZE) -> torch.Tensor:
    # frames: list of HxWx3 arrays in RGB
    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((size, size)),
        T.ToTensor(),                        # [3, H, W], in 0..1
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])
    # Shape to [T, 3, H, W]
    clip = torch.stack([tf(f) for f in frames], dim=0)
    # Reorder to [3, T, H, W] for 3D conv
    clip = clip.permute(1, 0, 2, 3)
    return clip

# -----------------------------
# Optical flow
# -----------------------------
def compute_flow_clip(frames: List[np.ndarray], size: int = FRAME_SIZE) -> torch.Tensor:
    """
    Compute dense optical flow between consecutive frames.
    Output as a 2-channel stack (u, v) per step, then pack as [2, T-1, H, W].
    """
    # Convert to gray and resize first for speed consistency
    gray = []
    for f in frames:
        f_rs = cv2.resize(f, (size, size))
        g = cv2.cvtColor(f_rs, cv2.COLOR_RGB2GRAY)
        gray.append(g)

    flows = []
    for i in range(len(gray) - 1):
        prev = gray[i]
        nxt  = gray[i + 1]
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # flow: HxW x 2 (u, v)
        flows.append(flow)

    if len(flows) == 0:
        # degenerate video of one frame, create zeros
        flows = [np.zeros((size, size, 2), dtype=np.float32)]

    # Stack along time: [T-1, H, W, 2]
    flow_arr = np.stack(flows, axis=0).astype(np.float32)

    # Normalize flow magnitude for stability
    # Optional: clip extreme values
    mag = np.linalg.norm(flow_arr, axis=-1, keepdims=True)
    eps = 1e-6
    flow_arr = flow_arr / (mag.mean() + eps)

    # To tensor shape [2, T-1, H, W]
    flow_t = torch.from_numpy(flow_arr).permute(3, 0, 1, 2)  # 2, T-1, H, W
    return flow_t

# -----------------------------
# Dataset
# -----------------------------
class VideoFolder(Dataset):
    def __init__(self, root_dir: str, num_frames: int = NUM_FRAMES, train: bool = True):
        self.root = root_dir
        self.num_frames = num_frames
        self.train = train
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            for vid in glob.glob(os.path.join(root_dir, cls, "*.mp4")):
                self.samples.append((vid, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = read_video_frames(path)
        if len(frames) == 0:
            # handle unreadable video by creating a dummy black clip
            frames = [np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)] * self.num_frames

        # sample frames
        inds = uniform_sample_indices(len(frames), self.num_frames)
        frames = [frames[i] for i in inds]

        rgb_clip = preprocess_rgb_clip(frames, FRAME_SIZE)            # [3, T, H, W]
        flow_clip = compute_flow_clip(frames, FRAME_SIZE)             # [2, T-1, H, W]

        return rgb_clip, flow_clip, label

# -----------------------------
# Simple 3D CNN blocks
# -----------------------------
def conv3x3x3(in_c, out_c, stride=(1,1,1)):
    return nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)

class Small3D(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, time_len=NUM_FRAMES):
        super().__init__()
        self.features = nn.Sequential(
            conv3x3x3(in_channels, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1,2,2)),

            conv3x3x3(32, 64),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),

            conv3x3x3(64, 128),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2,2,2)),
        )
        # temporal dimension after pooling: roughly time_len / 2
        self.gap = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, C, T, H, W]
        x = self.features(x)
        x = self.gap(x)               # [B, 128, 1, 1, 1]
        x = x.flatten(1)              # [B, 128]
        x = self.fc(x)                # [B, num_classes]
        return x

class TwoStreamNet(nn.Module):
    """
    RGB stream expects 3 channels.
    Flow stream expects 2 channels per time step. We feed [2, T-1, H, W].
    Shapes must be aligned on time before fusion. We will crop RGB time by 1 to match flow length.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.rgb = Small3D(in_channels=3, num_classes=num_classes)
        self.flow = Small3D(in_channels=2, num_classes=num_classes)

        # replace the last layers to expose penultimate features for fusion
        self.rgb.fc = nn.Identity()
        self.flow.fc = nn.Identity()

        self.classifier = nn.Linear(128 + 128, num_classes)

    def forward(self, rgb, flow):
        # rgb: [B, 3, T, H, W]
        # flow: [B, 2, T-1, H, W]
        # Align time by chopping the last RGB frame
        if rgb.shape[2] == flow.shape[2] + 1:
            rgb = rgb[:, :, :flow.shape[2], :, :]

        f_rgb = self.rgb.features(rgb)
        f_flow = self.flow.features(flow)

        f_rgb = self.rgb.gap(f_rgb).flatten(1)    # [B, 128]
        f_flow = self.flow.gap(f_flow).flatten(1) # [B, 128]

        fused = torch.cat([f_rgb, f_flow], dim=1) # [B, 256]
        out = self.classifier(fused)              # [B, num_classes]
        return out

# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, correct, running = 0, 0, 0.0
    for rgb, flow, y in loader:
        rgb = rgb.to(DEVICE)
        flow = flow.to(DEVICE)
        y = torch.tensor(y).to(DEVICE)

        optimizer.zero_grad()
        logits = model(rgb, flow)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * rgb.size(0)
        total += y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
    return running / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, correct, running = 0, 0, 0.0
    for rgb, flow, y in loader:
        rgb = rgb.to(DEVICE)
        flow = flow.to(DEVICE)
        y = torch.tensor(y).to(DEVICE)

        logits = model(rgb, flow)
        loss = criterion(logits, y)

        running += loss.item() * rgb.size(0)
        total += y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
    return running / total, correct / total

def make_loader(root, train: bool, batch_size=BATCH_SIZE):
    ds = VideoFolder(root, num_frames=NUM_FRAMES, train=train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True, drop_last=False)
    return ds, loader

def main():
    train_root = "data/train"
    val_root   = "data/val"

    train_ds, train_loader = make_loader(train_root, train=True)
    val_ds, val_loader     = make_loader(val_root, train=False)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    model = TwoStreamNet(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": train_ds.classes
    }, "checkpoints/asl_two_stream.pt")
    print("Saved to checkpoints/asl_two_stream.pt")

if __name__ == "__main__":
    main()
