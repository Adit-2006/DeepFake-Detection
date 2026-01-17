# src/train_streamed.py
# GPU-optimized streamed training WITH FPS = 12 sampling

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# ---------------- CONFIG ----------------
DATA_DIR = "data/videos"     # data/videos/real and data/videos/fake (recursive)
EPOCHS = 2
LR = 1e-4
BATCH_SIZE = 16
FPS = 12                    # <-- TRAINING FPS SET HERE
MODEL_PATH = "models/image_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

print(f"Training on device: {DEVICE}")

# Speedups
torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Model
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler()


def train_on_video(video_path: str, label: int):
    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(int(video_fps // FPS), 1)

    frames = []
    labels = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(transform(img))
            labels.append(label)

            if len(frames) == BATCH_SIZE:
                x = torch.stack(frames).to(DEVICE, non_blocking=True)
                y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

                optimizer.zero_grad()
                with autocast():
                    out = model(x)
                    loss = criterion(out, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                frames.clear()
                labels.clear()

        frame_id += 1

    # Train on remaining frames
    if frames:
        x = torch.stack(frames).to(DEVICE, non_blocking=True)
        y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    cap.release()


def train():
    model.train()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(DATA_DIR, cls)
            if not os.path.isdir(cls_dir):
                continue

            for root, _, files in os.walk(cls_dir):
                for fname in files:
                    if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                        video_path = os.path.join(root, fname)
                        print(f"Training on: {video_path}")
                        train_on_video(video_path, label)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
