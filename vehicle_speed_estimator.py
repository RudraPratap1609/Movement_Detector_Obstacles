
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv



DEFAULT_MODEL = "runs/detect/vehicle_detector/weights/best.pt"

CONFIDENCE_THRESHOLD = 0.3

AVG_VEHICLE_WIDTH_M = 1.8

TRACKER_CONFIG = dict(
    track_activation_threshold=0.25,
    lost_track_buffer=60,
    minimum_matching_threshold=0.8,
    frame_rate=30,
)

SPEED_SMOOTHING_WINDOW = 10

MAX_REASONABLE_SPEED_KMH = 200.0



class SpeedEstimator:

    def __init__(
        self,
        model_weights: str = DEFAULT_MODEL,
        avg_vehicle_width_m: float = AVG_VEHICLE_WIDTH_M,
        tracker_config: Optional[dict] = None,
    ):
        self.model = YOLO(model_weights)
        self.avg_vehicle_width_m = avg_vehicle_width_m

        cfg = tracker_config or TRACKER_CONFIG
        self.tracker = sv.ByteTrack(**cfg)

        self.track_state: Dict[int, dict] = defaultdict(
            lambda: {"prev_px": None, "prev_time": None, "speeds": []}
        )

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=1, text_scale=0.5, text_padding=5
        )

    @staticmethod
    def _bottom_center(bbox: np.ndarray) -> np.ndarray:
        """Bottom-center point of an (x1, y1, x2, y2) bbox."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, y2])

    @staticmethod
    def _bbox_width(bbox: np.ndarray) -> float:
        return float(bbox[2] - bbox[0])

    def _update_speed(
        self, track_id: int, pixel_point: np.ndarray, bbox_w: float, timestamp: float
    ) -> float:
        state = self.track_state[track_id]

        if state["prev_px"] is None:
            state["prev_px"] = pixel_point.copy()
            state["prev_time"] = timestamp
            return 0.0

        dt = timestamp - state["prev_time"]
        if dt <= 0:
            return self._smoothed_speed(track_id)

        displacement_px = float(np.linalg.norm(pixel_point - state["prev_px"]))

        pixels_per_meter = bbox_w / self.avg_vehicle_width_m
        if pixels_per_meter < 1.0:
            pixels_per_meter = 1.0  

        distance_m = displacement_px / pixels_per_meter
        speed_kmh = (distance_m / dt) * 3.6

        if speed_kmh > MAX_REASONABLE_SPEED_KMH:
            speed_kmh = self._smoothed_speed(track_id)

        state["speeds"].append(speed_kmh)
        if len(state["speeds"]) > SPEED_SMOOTHING_WINDOW:
            state["speeds"] = state["speeds"][-SPEED_SMOOTHING_WINDOW:]

        state["prev_px"] = pixel_point.copy()
        state["prev_time"] = timestamp

        return self._smoothed_speed(track_id)

    def _smoothed_speed(self, track_id: int) -> float:
        speeds = self.track_state[track_id]["speeds"]
        return float(np.mean(speeds)) if speeds else 0.0

    def _prune_stale_tracks(self, active_ids: set):
        stale = [tid for tid in self.track_state if tid not in active_ids]
        for tid in stale:
            del self.track_state[tid]

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> Tuple[np.ndarray, sv.Detections]:
        
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = detections.confidence >= CONFIDENCE_THRESHOLD
        detections = detections[mask]

        detections = self.tracker.update_with_detections(detections)

        labels: List[str] = []
        active_ids: set = set()

        if detections.tracker_id is not None:
            for bbox, tracker_id in zip(detections.xyxy, detections.tracker_id):
                tid = int(tracker_id)
                active_ids.add(tid)

                bc = self._bottom_center(bbox)
                bw = self._bbox_width(bbox)
                speed = self._update_speed(tid, bc, bw, timestamp)
                labels.append(f"#{tid} {speed:.1f} km/h")

        self._prune_stale_tracks(active_ids)

        annotated = frame.copy()
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        return annotated, detections


def run_pipeline(
    source_path: str,
    output_path: Optional[str] = None,
    display: bool = True,
    model_weights: str = DEFAULT_MODEL,
):

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Update tracker frame_rate to match video
    tracker_cfg = {**TRACKER_CONFIG, "frame_rate": int(fps)}
    estimator = SpeedEstimator(model_weights=model_weights, tracker_config=tracker_cfg)

    frame_idx = 0
    print(f"Processing {source_path}  ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)")
    print(f"Auto-calibration: using avg vehicle width = {AVG_VEHICLE_WIDTH_M} m")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            obstacle_y = height // 2
            obstacle_h = 50
            cv2.rectangle(
                frame,
                (0, obstacle_y - obstacle_h // 2),
                (width, obstacle_y + obstacle_h // 2),
                (0, 0, 255),
                -1,  
            )
            cv2.putText(
                frame, "OBSTACLE", (width // 2 - 80, obstacle_y + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            annotated, _ = estimator.process_frame(frame, timestamp)

            if writer:
                writer.write(annotated)

            if display:
                cv2.imshow("Vehicle Speed Estimator", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user (q).")
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames else 0
                print(f"  frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Done. Processed {frame_idx} frames.")
    if output_path:
        print(f"Output saved to: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect, track, and estimate vehicle speed from video."
    )
    parser.add_argument("--source", required=True, help="Path to input .mp4 video")
    parser.add_argument("--output", default=None, help="Path to save annotated .mp4")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to .pt weights")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview")
    args = parser.parse_args()

    run_pipeline(
        source_path=args.source,
        output_path=args.output,
        display=not args.no_display,
        model_weights=args.model,
    )
