# src/video_infer.py

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.preprocess import extract_frames
# IMPROVEMENT: Import model components directly to perform In-Memory inference
# This avoids writing thousands of temporary JPGs to disk (huge speedup)
from src.image_infer import model, transform, DEVICE, TEMPERATURE
from src.temporal_utils import temporal_localization


def infer_video(video_path: str) -> dict:
    """
    Video deepfake detection with:
    - face tracking
    - in-memory inference (optimized)
    - adaptive thresholds with safety floor
    - temporal localization
    """

    fps = 5
    frames, times = extract_frames(video_path, fps=fps)

    if not frames:
        return {
            "input_type": "video",
            "video_is_fake": False,
            "overall_confidence": 0.0,
            "manipulated_segments": []
        }

    probs = []

    # ---------------- OPTIMIZED INFERENCE ----------------
    # Switch model to eval mode
    model.eval()
    
    for frame in frames:
        # 1. Convert BGR (OpenCV) to RGB (PIL)
        # We do this in memory instead of saving to disk
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 2. Transform & Move to Device
        x = transform(pil_img).unsqueeze(0).to(DEVICE)

        # 3. Inference
        with torch.no_grad():
            logits = model(x)
            scaled_logits = logits / TEMPERATURE
            p = F.softmax(scaled_logits, dim=1)
        
        # We assume Class 1 is Fake (consistent with image_infer logic)
        fake_prob = float(p[0][1].item())
        probs.append(fake_prob)

    # ---------------- ADAPTIVE THRESHOLDS ----------------
    mean_p = float(np.mean(probs))
    std_p = float(np.std(probs))

    # FIXED: Added a "Safety Floor" of 0.65.
    # Previous logic allowed the threshold to drop to ~0.05 on real videos, 
    # causing the model to "hallucinate" fakes in perfectly real footage.
    high_th = max(0.65, min(0.99, mean_p + 1.5 * std_p))
    
    # Low threshold helps connect fragmented segments
    low_th = max(0.40, mean_p)

    # ---------------- TEMPORAL LOCALIZATION ----------------
    segments = temporal_localization(
        times,
        probs,
        fps=fps,
        high_th=high_th,
        low_th=low_th,
        min_segment_sec=1.0 # Ignore blips shorter than 1s
    )

    overall_conf = max(
        [seg["confidence"] for seg in segments],
        default=0.0
    )

    return {
        "input_type": "video",
        "video_is_fake": len(segments) > 0,
        "overall_confidence": round(overall_conf, 2),
        "manipulated_segments": segments
    }