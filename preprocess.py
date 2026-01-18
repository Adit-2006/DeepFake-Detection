# src/preprocess.py

import cv2

def iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Box format: (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


def extract_frames(video_path: str, fps: int = 5):
    """
    Extracts face crops while enforcing same-face temporal consistency.
    Returns:
        frames: list of face images (BGR numpy arrays)
        times:  list of timestamps (seconds)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 30 # Fallback safety
    
    interval = max(int(video_fps // fps), 1)

    frames = []
    times = []
    prev_face = None
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                # IMPROVEMENT: Sort faces by area (w * h) descending.
                # We always want the largest face (the main subject).
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

                x, y, w, h = faces[0]
                curr_face = (x, y, x + w, y + h)

                # Tracking logic:
                # If this is the first face, OR it overlaps significantly with the previous face
                if prev_face is None or iou(prev_face, curr_face) > 0.3:
                    prev_face = curr_face
                    
                    # Crop logic
                    face_crop = frame[y:y + h, x:x + w]
                    frames.append(face_crop)
                    
                    # Timestamp logic
                    times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)

        frame_id += 1

    cap.release()
    return frames, times