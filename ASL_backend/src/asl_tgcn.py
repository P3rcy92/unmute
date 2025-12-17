import os
import sys
import glob
import math
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# MediaPipe extraction
# -----------------------------
def mp_extract_landmarks_from_video(video_path, image_size=256, max_frames=96):
    """
    Returns landmarks as a float32 array of shape [T, V, C]:
    - T frames (time), V joints, C=3 (x, y, visibility)
    Uses MediaPipe Holistic: pose+hands. Face is omitted by default for simplicity.
    """
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    ok = True
    while ok:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Subsample uniformly to max_frames if needed
    def uniform_indices(n, k):
        if n <= k:
            return list(range(n)) + [n-1] * (k - n)
        step = n / k
        return [int(i * step) for i in range(k)]

    idxs = uniform_indices(len(frames), max_frames)
    frames = [frames[i] for i in idxs]

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )

    # We use 33 pose + 21 left hand + 21 right hand = 75 joints
    # Order: pose(33), left hand(21), right hand(21)
    V = 33 + 21 + 21
    T = len(frames)
    data = np.zeros((T, V, 3), dtype=np.float32)

    for t, img in enumerate(frames):
        h, w, _ = img.shape
        res = holistic.process(img)

        def add_lms(lms, start_idx, max_n):
            if lms is None:
                return
            n = min(len(lms.landmark), max_n)
            for i in range(n):
                x = lms.landmark[i].x
                y = lms.landmark[i].y
                v = lms.landmark[i].visibility if hasattr(lms.landmark[i], "visibility") else 1.0
                data[t, start_idx + i, 0] = x
                data[t, start_idx + i, 1] = y
                data[t, start_idx + i, 2] = v

        add_lms(res.pose_landmarks, 0, 33)
        add_lms(res.left_hand_landmarks, 33, 21)
        add_lms(res.right_hand_landmarks, 33 + 21, 21)

    holistic.close()

    # Mask out low visibility and interpolate small gaps if desired
    # Simple normalization: zero-center and scale to unit box based on pose hips and shoulders
    # Compute a reference scale using pose landmarks if available
    pose_xy = data[:, :33, :2]
    vis = data[:, :33, 2:3]
    valid = (vis > 0.5).astype(np.float32)
    # Use mean over valid points as center
    denom = np.maximum(valid.sum(axis=(1, 0, 2), keepdims=True), 1.0)
    center = (pose_xy * valid).sum(axis=(1, 0), keepdims=True) / denom
    data[:, :, :2] -= center  # center at origin

    # Compute scale as mean distance to center for valid pose points
    dists = np.linalg.norm(pose_xy - center, axis=-1, keepdims=True) * valid
    scale = dists.sum() / np.maximum(valid.sum(), 1.0)
    if scale <= 1e-6:
        scale = 1.0
    data[:, :, :2] /= float(scale)

    return data  # [T, V, 3]

def extract_dataset_to_npz(src_root, dst_root, max_frames=96):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in src_root.iterdir() if d.is_dir()])
    for split in ["train", "val"]:
        split_dir = src_root / split
        if not split_dir.exists():
            continue
        for cls in classes:
            in_dir = split_dir / cls
            if not in_dir.exists():
                continue
            out_dir = dst_root / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for mp4 in sorted(in_dir.glob("*.mp4")):
                npz_path = out_dir / (mp4.stem + ".npz")
                if npz_path.exists():
                    continue
                arr = mp_extract_landmarks_from_video(mp4, max_frames=max_frames)
                # Save as npz with key "x"
                np.savez_compressed(npz_path, x=arr)
                print("Saved", npz_path)

# -----------------------------
# Graph utilities
# -----------------------------
def build_adjacency(num_pose=33, num_hand=21):
    """
    Build a simple fixed adjacency for 33 pose + 21 left + 21 right.
    Edges based on MediaPipe topology. For brevity, we include chains and a few cross links.
    """
    V = num_pose + num_hand + num_hand
    A = np.zeros((V, V), dtype=np.float32)

    def connect_pairs(pairs):
        for i, j in pairs:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Pose chain (subset)
    pose_edges = [
        (11, 13), (13, 15), (12, 14), (14, 16),    # arms
        (23, 25), (25, 27), (24, 26), (26, 28),    # legs
        (11, 12), (23, 24), (11, 23), (12, 24),    # shoulders-hips
        (0, 11), (0, 12)                           # neck to shoulders
    ]
    connect_pairs(pose_edges)

    # Left hand local chain indices in MediaPipe are 0..20, map to global 33..53
    def hand_edges(base):
        chain = [(0,1),(1,2),(2,3),(3,4),    # thumb
                 (5,6),(6,7),(7,8),          # index
                 (9,10),(10,11),(11,12),     # middle
                 (13,14),(14,15),(15,16),    # ring
                 (17,18),(18,19),(19,20)]    # pinky
        return [(base+a, base+b) for a,b in chain]

    left_base = 33
    right_base = 33 + 21
    connect_pairs(hand_edges(left_base))
    connect_pairs(hand_edges(right_base))

    # Connect wrists to pose: pose 15,16 are hands in some schemas, in MediaPipe wrists are 15, 16 or 19, 20 for hands
    # MediaPipe pose wrist indices: 15 right, 16 left
    # Connect pose left wrist(15) to left hand base joint(33+0), right wrist(16) to right hand base(33+21+0)
    connect_pairs([(16, left_base + 0), (15, right_base + 0)])

    # Add self loops
    for i in range(V):
        A[i, i] = 1.0

    # Normalize A by degree
    D = np.diag(1.0 / np.sqrt(np.clip(A.sum(axis=1), 1.0, None)))
    A_norm = D @ A @ D
    return torch.from_numpy(A_norm)  # [V, V]

# -----------------------------
# TGCN blocks
# -----------------------------
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, bias=False):
        super().__init__()
        self.A = nn.Parameter(A.clone(), requires_grad=False)  # fixed topology here
        self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        # x: [B, C, T, V]
        # graph conv: X' = Theta * X * A
        x = self.theta(x)                                # [B, C_out, T, V]
        x = torch.einsum('bctv,vw->bctw', x, self.A)     # right-multiply by adjacency
        return x

class TCN(nn.Module):
    def __init__(self, channels, stride=1, kernel_size=9):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C, T, V]
        return self.net(x)

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, stride=1, residual=True):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, A)
        self.tcn = TCN(out_c, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_c == out_c and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_joints, num_classes, A):
        super().__init__()
        c1, c2, c3, c4 = 64, 128, 256, 256
        self.data_bn = nn.BatchNorm1d(num_joints * 3)

        self.layer1 = STGCNBlock(3, c1, A, residual=False)
        self.layer2 = STGCNBlock(c1, c2, A, stride=2)
        self.layer3 = STGCNBlock(c2, c3, A, stride=2)
        self.layer4 = STGCNBlock(c3, c4, A, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # average over time and joints
        self.fc = nn.Linear(c4, num_classes)

    def forward(self, x):
        # x: [B, T, V, C] where C=3
        # reorder to [B, C, T, V]
        x = x.permute(0, 3, 1, 2).contiguous()
        b, c, t, v = x.shape
        x = x.view(b, c * v, t)
        x = self.data_bn(x)
        x = x.view(b, c, t, v)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# -----------------------------
# Dataset for NPZ tensors
# -----------------------------
class SkeletonNPZ(Dataset):
    def __init__(self, root_npz):
        self.root = Path(root_npz)
        self.samples = []
        self.classes = []
        for split in ["train", "val"]:
            pass  # just to keep naming aligned

        # discover classes from train split
        train_dir = self.root / "train"
        self.classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}

        # gather samples for both splits based on constructor root
        for cls in self.classes:
            for npz in sorted((self.root / "split_placeholder" / cls).glob("*.npz")):
                self.samples.append((npz, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        x = np.load(path)["x"].astype(np.float32)  # [T, V, 3]
        # Optionally random temporal crop or jitter here
        return torch.from_numpy(x), y

def make_loader(npz_root, split, batch_size=8, num_workers=2):
    # monkey-patch split path for SkeletonNPZ
    ds = SkeletonNPZ.__new__(SkeletonNPZ)
    SkeletonNPZ.__init__ = lambda *args, **kwargs: None  # no-op
    ds.root = Path(npz_root)
    train_dir = ds.root / "train"
    ds.classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    ds.class_to_idx = {c:i for i,c in enumerate(ds.classes)}
    ds.samples = []
    for cls in ds.classes:
        for p in sorted((ds.root / split / cls).glob("*.npz")):
            ds.samples.append((p, ds.class_to_idx[cls]))

    def collate(batch):
        xs, ys = zip(*batch)
        # Pad or truncate to the max T in batch
        T_max = max(x.shape[0] for x in xs)
        V = xs[0].shape[1]
        C = xs[0].shape[2]
        xb = torch.zeros((len(xs), T_max, V, C), dtype=torch.float32)
        for i, x in enumerate(xs):
            T = x.shape[0]
            xb[i, :T] = x
        yb = torch.tensor(ys, dtype=torch.long)
        return xb, yb

    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"),
                        num_workers=num_workers, collate_fn=collate, drop_last=False)
    return ds, loader

# -----------------------------
# Training
# -----------------------------
def train_tgcn(npz_root, epochs=10, lr=1e-3, batch_size=8, num_workers=2):
    # Build adjacency
    A = build_adjacency()
    V = A.shape[0]

    # Data
    train_ds, train_loader = make_loader(npz_root, "train", batch_size, num_workers)
    val_ds, val_loader = make_loader(npz_root, "val", batch_size, num_workers)

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STGCN(num_joints=V, num_classes=num_classes, A=A.to(device)).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)           # [B, C]
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

            loss_sum += loss.item() * xb.size(0)
            tot += xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        tr_loss = loss_sum / max(tot,1)
        tr_acc = correct / max(tot,1)

        # val
        model.eval()
        tot, correct, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_sum += loss.item() * xb.size(0)
                tot += xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
        va_loss = loss_sum / max(tot,1)
        va_acc = correct / max(tot,1)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "classes": train_ds.classes
    }, "checkpoints/tgcn_skeleton.pt")
    print("Saved to checkpoints/tgcn_skeleton.pt")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "train"])
    ap.add_argument("--src_videos", type=str, default="data")          # input videos root
    ap.add_argument("--out_npz", type=str, default="npz")              # output npz root
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=96)
    args = ap.parse_args()

    if args.mode == "extract":
        # Read videos from data/train and data/val, write npz/train and npz/val
        extract_dataset_to_npz(args.src_videos, args.out_npz, max_frames=args.max_frames)
    else:
        train_tgcn(args.out_npz, epochs=args.epochs, lr=args.lr, batch_size=args.batch)

if __name__ == "__main__":
    main()
