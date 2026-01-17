# src/video_infer.py

import cv2
import numpy as np
import tempfile

from src.preprocess import extract_frames
from src.image_infer import infer_image
from src.temporal_utils import temporal_localization


def infer_video(video_path: str) -> dict:
    """
    Video deepfake detection with:
    - face tracking
    - adaptive thresholds
    - temporal localization
    """

    fps = 5
    frames, times = extract_frames(video_path, fps=fps)

    probs = []

    for frame in frames:
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            cv2.imwrite(tmp.name, frame)
            result = infer_image(tmp.name)
            probs.append(result["confidence"])

    if len(probs) == 0:
        return {
            "input_type": "video",
            "video_is_fake": False,
            "overall_confidence": 0.0,
            "manipulated_segments": []
        }

    # -------- Adaptive Thresholds --------
    mean_p = float(np.mean(probs))
    std_p = float(np.std(probs))

    high_th = min(0.9, mean_p + 1.5 * std_p)
    low_th = max(0.3, mean_p)

    # -------- Temporal Localization --------
    segments = temporal_localization(
        times,
        probs,
        fps=fps,
        high_th=high_th,
        low_th=low_th
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