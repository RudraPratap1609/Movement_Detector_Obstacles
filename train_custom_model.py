"""
Train YOLOv8 on Local Vehicle Dataset
======================================
Trains a YOLOv8 model on the local Vehicles/ dataset (YOLO format).

Usage:
    python train_custom_model.py
    python train_custom_model.py --epochs 50 --batch 8 --device cpu

After training, best weights are saved to:
    runs/detect/vehicle_detector/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DATASET_YAML = str(Path(__file__).parent / "Vehicles" / "data.yaml")
BASE_MODEL = "yolov8n.pt"
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "0"  # "0" for first GPU, "cpu" for CPU-only
PROJECT_NAME = "runs"
RUN_NAME = "vehicle_detector"


def train(
    data_yaml: str = DATASET_YAML,
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    imgsz: int = IMG_SIZE,
    batch: int = BATCH_SIZE,
    device: str = DEVICE,
) -> str:
    """Fine-tune YOLOv8 on the local dataset. Returns path to best weights."""

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    model = YOLO(base_model)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        # ── Augmentation (tuned for vehicle detection) ────────────────
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        flipud=0.0,
        # ── Training strategy ─────────────────────────────────────────
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
    )

    weights_path = Path(PROJECT_NAME) / RUN_NAME / "weights" / "best.pt"
    print(f"\nTraining complete! Best weights: {weights_path}")
    return str(weights_path)


def validate(weights_path: str, data_yaml: str = DATASET_YAML):
    """Run validation and print metrics."""
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=IMG_SIZE, device=DEVICE)

    print(f"\n{'='*50}")
    print(f"  Validation Results")
    print(f"{'='*50}")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 vehicle detector")
    parser.add_argument("--data", default=DATASET_YAML, help="Path to data.yaml")
    parser.add_argument("--model", default=BASE_MODEL, help="Base model (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE)
    parser.add_argument("--device", default=DEVICE, help="'0' for GPU, 'cpu' for CPU")
    args = parser.parse_args()

    print("=" * 60)
    print("  Vehicle Detector — Training")
    print("=" * 60)

    # Train
    print(f"\n[1/2] Training on {args.data} ...")
    weights = train(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # Validate
    print(f"\n[2/2] Validating ...")
    validate(weights, args.data)

    print(f"\nTo run inference:")
    print(f'  python vehicle_speed_estimator.py --source video.mp4 --model "{weights}"')
