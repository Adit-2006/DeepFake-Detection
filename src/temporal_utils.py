# src/temporal_utils.py
# Full temporal reasoning utilities (RESTORED + IMPROVED)

from typing import List, Dict
import numpy as np


# --------------------------------------------------
# Utility: seconds → MM:SS
# --------------------------------------------------
def seconds_to_timestamp(sec: float) -> str:
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes:02d}:{seconds:02d}"


# --------------------------------------------------
# Core Temporal Localization Logic
# --------------------------------------------------
def temporal_localization(
    frame_times: List[float],
    frame_probs: List[float],
    fps: int = 5,
    window_size: int = 5,          # frames per window (~1s)
    high_th: float = 0.7,          # enter fake segment
    low_th: float = 0.4,           # exit fake segment
    min_segment_sec: float = 1.0   # discard tiny segments
) -> List[Dict]:
    """
    Performs temporal reasoning to localize manipulated segments.

    Pipeline:
    1. Frame probabilities → sliding windows
    2. Robust window scoring (median)
    3. Hysteresis thresholding
    4. Segment aggregation
    5. Stable segment confidence computation

    Returns:
        List of segments with start_time, end_time, confidence
    """

    if len(frame_probs) < window_size:
        return []

    # --------------------------------------------------
    # 1. Sliding Window Aggregation
    # --------------------------------------------------
    windows = []
    for i in range(len(frame_probs) - window_size + 1):
        w_probs = frame_probs[i:i + window_size]
        score = float(np.median(w_probs))  # robust to noise

        windows.append({
            "start": frame_times[i],
            "end": frame_times[i + window_size - 1],
            "score": score
        })

    # --------------------------------------------------
    # 2. Hysteresis-Based Segment Detection
    # --------------------------------------------------
    segments = []
    in_fake = False
    seg_start = None
    seg_scores = []

    for w in windows:
        if not in_fake:
            # Enter fake region
            if w["score"] >= high_th:
                in_fake = True
                seg_start = w["start"]
                seg_scores = [w["score"]]
        else:
            # Stay in fake region
            if w["score"] >= low_th:
                seg_scores.append(w["score"])
            else:
                # Exit fake region
                seg_end = w["end"]
                duration = seg_end - seg_start

                if duration >= min_segment_sec:
                    seg_scores_np = np.array(seg_scores)

                    # --------------------------------------------------
                    # 3. Stable Segment Confidence (CORE FUNCTIONALITY)
                    # --------------------------------------------------
                    # - Top-k mean → strength
                    # - Std penalty → stability
                    k = max(1, int(0.7 * len(seg_scores_np)))
                    strength = float(np.mean(np.sort(seg_scores_np)[-k:]))
                    stability = float(1.0 - np.std(seg_scores_np))
                    confidence = max(0.0, strength * stability)

                    segments.append({
                        "start_time": seconds_to_timestamp(seg_start),
                        "end_time": seconds_to_timestamp(seg_end),
                        "confidence": round(confidence, 3)
                    })

                in_fake = False
                seg_start = None
                seg_scores = []

    # --------------------------------------------------
    # 4. Handle Open Segment at End of Video
    # --------------------------------------------------
    if in_fake and seg_start is not None:
        seg_end = windows[-1]["end"]
        duration = seg_end - seg_start

        if duration >= min_segment_sec:
            seg_scores_np = np.array(seg_scores)
            k = max(1, int(0.7 * len(seg_scores_np)))
            strength = float(np.mean(np.sort(seg_scores_np)[-k:]))
            stability = float(1.0 - np.std(seg_scores_np))
            confidence = max(0.0, strength * stability)

            segments.append({
                "start_time": seconds_to_timestamp(seg_start),
                "end_time": seconds_to_timestamp(seg_end),
                "confidence": round(confidence, 3)
            })

    return segments
