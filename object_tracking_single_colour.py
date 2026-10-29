import argparse
from collections import deque

import cv2
import numpy as np

WINDOW_NAME = "Automatic Colour Detection and Tracking"
FRAME_SIZE = (640, 480)
MIN_OBJECT_AREA = 900
MAX_OBJECT_FRACTION = 0.45
TRAIL_LENGTH = 20
MAX_MISSED_FRAMES = 6
SMOOTHING = 0.65

# OpenCV HSV hue ranges. Black, white and grey are handled by saturation/value,
# because these colours do not have a reliable hue.
CHROMATIC_RANGES = [
    ("Red",    (0, 75, 45),    (10, 255, 255)),
    ("Red",    (170, 75, 45), (179, 255, 255)),
    ("Orange", (11, 75, 45),  (24, 255, 255)),
    ("Yellow", (25, 75, 60),  (39, 255, 255)),
    ("Green",  (40, 65, 40),  (80, 255, 255)),
    ("Cyan",   (81, 65, 40),  (99, 255, 255)),
    ("Blue",   (100, 65, 40), (130, 255, 255)),
    ("Purple", (131, 55, 35), (154, 255, 255)),
    ("Pink",   (155, 40, 80), (169, 255, 255)),
    ("Brown",  (5, 45, 25),   (24, 200, 175)),
]


def make_masks(hsv):
    """Return separate masks so one colour cannot swallow all other colours."""
    masks = []
    for name, lower, upper in CHROMATIC_RANGES:
        mask = cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        )
        masks.append((name, mask))

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Neutral colours: use saturation and brightness, not hue.
    masks.extend([
        ("Black", cv2.inRange(hsv, np.array([0, 0, 0], np.uint8),
                              np.array([179, 255, 55], np.uint8))),
        ("White", cv2.inRange(hsv, np.array([0, 0, 205], np.uint8),
                              np.array([179, 55, 255], np.uint8))),
        ("Grey",  cv2.inRange(hsv, np.array([0, 0, 56], np.uint8),
                              np.array([179, 55, 204], np.uint8))),
    ])
    return masks


def clean_mask(mask):
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def detect_regions(frame):
    # Slight blur reduces sensor noise while retaining useful object boundaries.
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    max_area = height * width * MAX_OBJECT_FRACTION
    found = []

    for colour_name, raw_mask in make_masks(hsv):
        mask = clean_mask(raw_mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_OBJECT_AREA or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 22 or h < 22:
                continue
            # Ignore regions that touch the image border; these are often background.
            if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                continue

            cx = x + w // 2
            cy = y + h // 2
            found.append({
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "colour": colour_name,
                "area": area,
            })

    # Different masks can find the same region. Keep the best single label for it.
    found.sort(key=lambda d: d["area"], reverse=True)
    unique = []
    for candidate in found:
        if any(box_iou(candidate["bbox"], item["bbox"]) > 0.50 for item in unique):
            continue
        unique.append(candidate)
    # Show only the largest meaningful colour region, so the camera view stays clean.
    # The detector still checks all colours, but it draws just one result at a time.
    return unique[:1]


class SimpleTracker:
    """Match detections to nearby previous centres and smooth small frame-to-frame jumps."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        for track in self.tracks.values():
            track["missed"] += 1

        used = set()
        for detection in detections:
            cx, cy = detection["center"]
            best_id, best_distance = None, 70.0

            for track_id, track in self.tracks.items():
                if track_id in used or track["colour"] != detection["colour"]:
                    continue
                px, py = track["center"]
                distance = float(np.hypot(cx - px, cy - py))
                if distance < best_distance:
                    best_id, best_distance = track_id, distance

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {
                    "colour": detection["colour"],
                    "center": (float(cx), float(cy)),
                    "bbox": detection["bbox"],
                    "missed": 0,
                    "trail": deque(maxlen=TRAIL_LENGTH),
                }
            else:
                track = self.tracks[best_id]
                px, py = track["center"]
                # Smooth coordinates to reduce visual jitter.
                smooth_x = SMOOTHING * px + (1 - SMOOTHING) * cx
                smooth_y = SMOOTHING * py + (1 - SMOOTHING) * cy
                track["center"] = (smooth_x, smooth_y)
                track["bbox"] = detection["bbox"]
                track["missed"] = 0
                track["colour"] = detection["colour"]

            track = self.tracks[best_id]
            track["trail"].append((int(track["center"][0]), int(track["center"][1])))
            detection["track_id"] = best_id
            detection["center"] = (int(track["center"][0]), int(track["center"][1]))
            detection["trail"] = track["trail"]
            used.add(best_id)

        # Keep tracks briefly through missed frames to avoid flickering.
        for track_id in list(self.tracks):
            if self.tracks[track_id]["missed"] > MAX_MISSED_FRAMES:
                del self.tracks[track_id]
        return detections


COLOUR_BGR = {
    "Red": (0, 0, 255), "Orange": (0, 140, 255), "Yellow": (0, 255, 255),
    "Green": (0, 220, 0), "Cyan": (255, 255, 0), "Blue": (255, 80, 0),
    "Purple": (220, 0, 220), "Pink": (203, 192, 255), "Brown": (42, 42, 165),
    "Black": (40, 40, 40), "White": (255, 255, 255), "Grey": (150, 150, 150),
}


def draw_results(frame, detections):
    for detection in detections:
        x, y, w, h = detection["bbox"]
        cx, cy = detection["center"]
        colour = detection["colour"]
        draw_colour = COLOUR_BGR.get(colour, (255, 255, 255))

        trail = list(detection["trail"])
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], draw_colour, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x + w, y + h), draw_colour, 2)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
        label = f"Color: {colour} | Center: ({cx}, {cy})"
        text_y = max(24, y - 8)
        # Small dark backing keeps the label readable without a central overlay.
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x, text_y - th - 5), (x + tw + 6, text_y + 3), (25, 25, 25), -1)
        cv2.putText(frame, label, (x + 3, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, draw_colour, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Automatic colour detection and tracking")
    parser.add_argument("--video", default=None, help="Optional video file instead of webcam")
    args = parser.parse_args()

    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera/video source: {source}")

    tracker = SimpleTracker()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 700)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, FRAME_SIZE)
            detections = tracker.update(detect_regions(frame))
            draw_results(frame, detections)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
