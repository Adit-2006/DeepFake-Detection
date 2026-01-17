# src/image_infer.py

import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Temperature scaling parameter (fixed)
TEMPERATURE = 2.0

# Decision threshold for images
IMAGE_FAKE_THRESHOLD = 0.75

# ---------------- Model ----------------
model = resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/image_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- Preprocess ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def infer_image(image_path: str) -> dict:
    """
    Image deepfake inference with calibrated confidence and decision.
    """

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        scaled_logits = logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=1)

    fake_prob = float(probs[0][1].item())
    is_fake = fake_prob >= IMAGE_FAKE_THRESHOLD

    return {
        "input_type": "image",
        "is_fake": is_fake,
        "confidence": round(fake_prob, 3)
    }