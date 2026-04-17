import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import time

# ══════════════════════════════════════════
#  Classes Config
# ══════════════════════════════════════════
CLASS_NAMES = {
    0: "Gloves",
    1: "Helmet",
    2: "No-Gloves",
    3: "No-Helmet",
    4: "No-Shoes",
    5: "No-Vest",
    6: "Shoes",
    7: "Vest",
}


VIOLATION_CLASSES = {2, 3, 4, 5}  # No-Gloves, No-Helmet, No-Shoes, No-Vest
SAFE_CLASSES     = {0, 1, 6, 7}   # Gloves, Helmet, Shoes, Vest

# Colors (BGR)
COLORS = {
    "safe":      (0, 200, 80),    # Green
    "violation": (0, 50, 220),    # Red
    "mask_safe": (0, 200, 80, 80),
    "mask_viol": (0, 50, 220, 80),
    "text_bg":   (20, 20, 20),
    "overlay":   (0, 0, 0),
}

# Severity levels
SEVERITY = {
    3: "CRITICAL",   # No-Helmet
    5: "HIGH",       # No-Vest
    2: "MEDIUM",     # No-Gloves
    4: "MEDIUM",     # No-Shoes
}


# ══════════════════════════════════════════
#  Detection Result Dataclass
# ══════════════════════════════════════════
@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       list          # [x1, y1, x2, y2]
    mask:       Optional[np.ndarray] = None
    is_violation: bool = False
    severity:   str = "INFO"
    timestamp:  float = field(default_factory=time.time)


@dataclass
class FrameResult:
    detections:       list
    violations:       list
    compliance_score: float
    annotated_frame:  np.ndarray
    fps:              float
    timestamp:        float = field(default_factory=time.time)


# ══════════════════════════════════════════
#  PPE Detector
# ══════════════════════════════════════════
class PPEDetector:
    def __init__(self, model_path: str, conf: float = 0.45, iou: float = 0.5):
        """
        Args:
            model_path : path to best.pt (YOLO26-seg)
            conf       : confidence threshold
            iou        : IoU threshold for NMS
        """
        print(f"🔄 Loading model from: {model_path}")
        self.model      = YOLO(model_path)
        self.conf       = conf
        self.iou        = iou
        self._prev_time = time.time()
        self.fps        = 0.0
        print("✅ Model loaded successfully!")
        print(f"   Classes: {CLASS_NAMES}")

    # ── FPS ───────────────────────────────
    def _update_fps(self):
        now           = time.time()
        self.fps      = 1.0 / max(now - self._prev_time, 1e-6)
        self._prev_time = now

    # ── Compliance Score ──────────────────
    def _calc_compliance(self, detections: list) -> float:
        if not detections:
            return 100.0
        violations = sum(1 for d in detections if d.is_violation)
        total      = len(detections)
        return round((1 - violations / total) * 100, 1)

    # ── Draw Mask ─────────────────────────
    def _draw_mask(self, frame: np.ndarray, mask: np.ndarray,
                   is_violation: bool, alpha: float = 0.35):
        color = (0, 50, 220) if is_violation else (0, 200, 80)
        overlay = frame.copy()
        if mask is not None and mask.shape == frame.shape[:2]:
            overlay[mask > 0] = color
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Draw BBox + Label ─────────────────
    def _draw_box(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = map(int, det.bbox)
        color = COLORS["violation"] if det.is_violation else COLORS["safe"]
        thickness = 2

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label background
        label = f"{det.class_name} {det.confidence:.0%}"
        if det.is_violation:
            label = f"⚠ {label}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Draw HUD ──────────────────────────
    def _draw_hud(self, frame: np.ndarray, result: FrameResult):
        h, w = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 46), (20, 20, 20), -1)

        # Compliance color
        score = result.compliance_score
        if score >= 80:
            sc_color = (0, 200, 80)
        elif score >= 50:
            sc_color = (0, 165, 255)
        else:
            sc_color = (0, 50, 220)


        cv2.putText(frame, f"PPE Monitor", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Compliance: {score:.0f}%", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, sc_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {result.fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)

        # Violations bar (bottom)
        if result.violations:
            vnames = ", ".join(set(d.class_name for d in result.violations))
            bar_text = f"VIOLATIONS: {vnames}"
            cv2.rectangle(frame, (0, h - 36), (w, h), (0, 30, 160), -1)
            cv2.putText(frame, bar_text, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Main Process Frame ─────────────────
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        self._update_fps()
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            stream=False,
        )

        detections = []
        annotated  = frame.copy()

        for r in results:
            boxes = r.boxes
            masks = r.masks  # None if detection-only model

            for i, box in enumerate(boxes):
                cls_id     = int(box.cls[0])
                cls_name   = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                conf_val   = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                is_viol    = cls_id in VIOLATION_CLASSES
                severity   = SEVERITY.get(cls_id, "INFO")

                # Segmentation mask
                seg_mask = None
                if masks is not None and i < len(masks.data):
                    seg_mask = masks.data[i].cpu().numpy().astype(np.uint8)
                    # Resize mask to frame size
                    seg_mask = cv2.resize(seg_mask,
                                          (frame.shape[1], frame.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                det = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf_val,
                    bbox=[x1, y1, x2, y2],
                    mask=seg_mask,
                    is_violation=is_viol,
                    severity=severity,
                )

                # Neglect helmet if up, shoes/no-shoes if down
                center_y = (det.bbox[1] + det.bbox[3]) / 2
                if cls_id == 1 and center_y < frame.shape[0] * 0.4:
                    continue
                if cls_id in {4, 6} and center_y > frame.shape[0] * 0.6:
                    continue

                detections.append(det)

                # Draw mask first (under box)
                if seg_mask is not None:
                    self._draw_mask(annotated, seg_mask, is_viol)

                # Draw bounding box + label
                self._draw_box(annotated, det)

        violations       = [d for d in detections if d.is_violation]
        compliance_score = self._calc_compliance(detections)

        frame_result = FrameResult(
            detections=detections,
            violations=violations,
            compliance_score=compliance_score,
            annotated_frame=annotated,
            fps=round(self.fps, 1),
        )

        self._draw_hud(annotated, frame_result)
        return frame_result

    # ── Process Video File ────────────────
    def process_video(self, video_path: str, output_path: str = None,
                      callback=None):
        """
        Process a video file frame by frame.
        callback(frame_result) called each frame — use for DB logging.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        print(f"🎬 Processing video: {video_path}")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)
            frame_count += 1

            if writer:
                writer.write(result.annotated_frame)

            if callback:
                callback(result)

            if frame_count % 30 == 0:
                print(f"   Frame {frame_count} | "
                      f"Compliance: {result.compliance_score}% | "
                      f"Violations: {len(result.violations)}")

        cap.release()
        if writer:
            writer.release()

        print(f"✅ Done! Processed {frame_count} frames.")
        return frame_count

    # ── Process Webcam ────────────────────
    def process_webcam(self, cam_index: int = 0, callback=None):
        """
        Live webcam processing.
        callback(frame_result) called each frame — use for DB logging.
        Returns generator of FrameResult for Streamlit.
        """
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam index: {cam_index}")

        print(f"📷 Webcam started (index={cam_index})")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_frame(frame)

                if callback:
                    callback(result)

                yield result

        finally:
            cap.release()
            print("📷 Webcam released.")

    # ── Process Single Image ──────────────
    def process_image(self, image_path: str) -> FrameResult:
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.process_frame(frame)
