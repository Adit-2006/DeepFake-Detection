# src/train_streamed.py

import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# ---------------- CONFIG ----------------
VIDEO_DIR = "data/videos"    # data/videos/real, data/videos/fake
IMAGE_DIR = "data/images"    # data/images/real, data/images/fake
EPOCHS = 5
LR = 5e-5
BATCH_SIZE = 16
FPS = 12
INPUT_SIZE = 160
MODEL_PATH = "models/image_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

print(f"Training on device: {DEVICE}")

torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)
torch.set_num_threads(os.cpu_count())

# ---------------- PREPROCESS ----------------
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor()
])

# ---------------- MODEL ----------------
model = resnet18(weights="IMAGENET1K_V1")

# Freeze backbone
for p in model.parameters():
    p.requires_grad = False

# Unfreeze last block + classifier
for p in model.layer4.parameters():
    p.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)
for p in model.fc.parameters():
    p.requires_grad = True

model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scaler = GradScaler()


# ---------------- IMAGE TRAINING ----------------
def train_on_images(cls_dir: str, label: int):
    frames, labels = [], []

    for root, _, files in os.walk(cls_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, fname)
                try:
                    with torch.no_grad():
                        img = Image.open(img_path).convert("RGB")
                        frames.append(transform(img))
                        labels.append(label)
                except Exception:
                    continue

                if len(frames) == BATCH_SIZE:
                    x = torch.stack(frames).pin_memory().to(DEVICE, non_blocking=True)
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

    if frames:
        x = torch.stack(frames).pin_memory().to(DEVICE, non_blocking=True)
        y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


# ---------------- VIDEO TRAINING ----------------
def train_on_video(video_path: str, label: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = FPS

    interval = max(int(video_fps // FPS), 1)

    frames, labels = [], []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            with torch.no_grad():
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(transform(img))
                labels.append(label)

            if len(frames) == BATCH_SIZE:
                x = torch.stack(frames).pin_memory().to(DEVICE, non_blocking=True)
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

    if frames:
        x = torch.stack(frames).pin_memory().to(DEVICE, non_blocking=True)
        y = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    cap.release()


# ---------------- TRAIN LOOP ----------------
def train():
    model.train()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for label, cls in enumerate(["real", "fake"]):

            # ---- Train on images ----
            img_dir = os.path.join(IMAGE_DIR, cls)
            if os.path.isdir(img_dir):
                print(f"Training on images: {img_dir}")
                train_on_images(img_dir, label)

            # ---- Train on videos ----
            vid_dir = os.path.join(VIDEO_DIR, cls)
            if os.path.isdir(vid_dir):
                for root, _, files in os.walk(vid_dir):
                    for fname in files:
                        if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                            video_path = os.path.join(root, fname)
                            print(f"Training on video: {video_path}")
                            train_on_video(video_path, label)

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Checkpoint saved after epoch {epoch + 1}")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
