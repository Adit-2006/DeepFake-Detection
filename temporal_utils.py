# src/temporal_utils.py

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
    Temporal localization using robust sliding window smoothing.
    
    FIXED: 
    - Removed 'video_mean' mixing (prevents suppression of short fakes).
    - Removed 'global_consistency' penalty (prevents penalizing mixed videos).
    """

    if len(frame_probs) < window_size:
        return []

    # --------------------------------------------------
    # Sliding window smoothing (Median Filter)
    # --------------------------------------------------
    # We use median to remove random single-frame spikes (noise)
    smoothed_probs = []
    
    # Pad the start to align indices
    pad = window_size // 2
    
    for i in range(len(frame_probs)):
        # Define window boundaries
        start = max(0, i - pad)
        end = min(len(frame_probs), i + pad + 1)
        
        window = frame_probs[start:end]
        smoothed_probs.append(float(np.median(window)))

    # --------------------------------------------------
    # Hysteresis Thresholding
    # --------------------------------------------------
    segments = []
    in_fake = False
    seg_start_idx = None
    
    for i, score in enumerate(smoothed_probs):
        curr_time = frame_times[i]

        if not in_fake:
            # Trigger: Score must exceed High Threshold
            if score >= high_th:
                in_fake = True
                seg_start_idx = i
        else:
            # Sustain: Keep segment alive as long as it stays above Low Threshold
            if score < low_th:
                # Segment ended
                seg_end_idx = i - 1
                duration = frame_times[seg_end_idx] - frame_times[seg_start_idx]

                if duration >= min_segment_sec:
                    # Extract raw probabilities for this segment to calculate confidence
                    raw_segment_probs = frame_probs[seg_start_idx : seg_end_idx + 1]
                    
                    # Robust Confidence: Mean of the top 50% of predictions in this segment
                    # This avoids being dragged down by edge-transition frames
                    k = max(1, int(0.5 * len(raw_segment_probs)))
                    confidence = np.mean(np.sort(raw_segment_probs)[-k:])

                    segments.append({
                        "start_time": seconds_to_timestamp(frame_times[seg_start_idx]),
                        "end_time": seconds_to_timestamp(frame_times[seg_end_idx]),
                        "confidence": round(float(confidence), 3)
                    })

                in_fake = False
                seg_start_idx = None

    # --------------------------------------------------
    # Handle open segment at end of video
    # --------------------------------------------------
    if in_fake and seg_start_idx is not None:
        seg_end_idx = len(frame_probs) - 1
        duration = frame_times[seg_end_idx] - frame_times[seg_start_idx]

        if duration >= min_segment_sec:
            raw_segment_probs = frame_probs[seg_start_idx : seg_end_idx + 1]
            k = max(1, int(0.5 * len(raw_segment_probs)))
            confidence = np.mean(np.sort(raw_segment_probs)[-k:])

            segments.append({
                "start_time": seconds_to_timestamp(frame_times[seg_start_idx]),
                "end_time": seconds_to_timestamp(frame_times[seg_end_idx]),
                "confidence": round(float(confidence), 3)
            })

    return segments