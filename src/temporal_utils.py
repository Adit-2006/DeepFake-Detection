# src/temporal_utils.py
# Temporal reasoning with VIDEO-LEVEL PERSISTENT MEMORY

from typing import List, Dict
import numpy as np


def seconds_to_timestamp(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def temporal_localization(
    frame_times: List[float],
    frame_probs: List[float],
    fps: int = 5,
    window_size: int = 5,
    high_th: float = 0.7,
    low_th: float = 0.4,
    min_segment_sec: float = 1.0
) -> List[Dict]:
    """
    Temporal localization with persistent VIDEO-LEVEL memory.
    Memory resets per video, but persists for entire video duration.
    """

    if len(frame_probs) < window_size:
        return []

    # --------------------------------------------------
    # Video-level persistent memory (global context)
    # --------------------------------------------------
    video_mean = 0.0
    video_var = 0.0
    seen = 0

    # --------------------------------------------------
    # Sliding window scoring
    # --------------------------------------------------
    windows = []
    for i in range(len(frame_probs) - window_size + 1):
        w_probs = frame_probs[i:i + window_size]
        score = float(np.median(w_probs))

        # Update video-level running statistics
        seen += 1
        delta = score - video_mean
        video_mean += delta / seen
        video_var += delta * (score - video_mean)

        windows.append({
            "start": frame_times[i],
            "end": frame_times[i + window_size - 1],
            "score": score
        })

    video_std = float(np.sqrt(video_var / max(seen, 1)))

    # --------------------------------------------------
    # Hysteresis + persistent memory
    # --------------------------------------------------
    segments = []
    in_fake = False
    seg_start = None
    seg_scores = []

    for w in windows:
        # Memory-adjusted score
        adjusted_score = 0.7 * w["score"] + 0.3 * video_mean

        if not in_fake:
            if adjusted_score >= high_th:
                in_fake = True
                seg_start = w["start"]
                seg_scores = [adjusted_score]
        else:
            if adjusted_score >= low_th:
                seg_scores.append(adjusted_score)
            else:
                seg_end = w["end"]
                duration = seg_end - seg_start

                if duration >= min_segment_sec:
                    scores = np.array(seg_scores)

                    k = max(1, int(0.7 * len(scores)))
                    strength = float(np.mean(np.sort(scores)[-k:]))
                    stability = float(1.0 - np.std(scores))
                    global_consistency = float(1.0 - video_std)

                    confidence = max(
                        0.0,
                        strength * stability * global_consistency
                    )

                    segments.append({
                        "start_time": seconds_to_timestamp(seg_start),
                        "end_time": seconds_to_timestamp(seg_end),
                        "confidence": round(confidence, 3)
                    })

                in_fake = False
                seg_start = None
                seg_scores = []

    # --------------------------------------------------
    # Handle open segment at end of video
    # --------------------------------------------------
    if in_fake and seg_start is not None:
        seg_end = windows[-1]["end"]
        duration = seg_end - seg_start

        if duration >= min_segment_sec:
            scores = np.array(seg_scores)
            k = max(1, int(0.7 * len(scores)))
            strength = float(np.mean(np.sort(scores)[-k:]))
            stability = float(1.0 - np.std(scores))
            global_consistency = float(1.0 - video_std)

            confidence = max(
                0.0,
                strength * stability * global_consistency
            )

            segments.append({
                "start_time": seconds_to_timestamp(seg_start),
                "end_time": seconds_to_timestamp(seg_end),
                "confidence": round(confidence, 3)
            })

    return segments
