#!/usr/bin/env python3
"""
Road Obstacle & Anomaly Segmentation Pipeline using SAM 3
==========================================================

Strategy:
  1. Drivable Space: Segment "road" to define the active zone.
  2. Traffic Sweep: Segment known vehicles.
  3. Anomaly Sweep: Segment broad list of potential obstructions/debris.
  4. Road Proximity Filter: Keep only objects touching/overlapping the road.
  5. Spatial Subtraction: Suppress any obstruction that heavily overlaps a vehicle.
  6. Deduplication & Clean Color Visualisation: Render only isolated anomalies without label clutter.

Usage:
    # Process a whole directory (e.g. dir 3):
    python Upgraded_segmentation.py --input "dir 3" --output-dir "output_dir3"

    # Process a single image:
    python Upgraded_segmentation.py --input "/path/to/image.jpg" --output "result.jpg"
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ROAD_PHRASES: list[str] = [
    "road"
]

# Step 2: Traffic Sweep
VEHICLE_PHRASES: list[str] = [
    "car", 
    "truck", 
    "motorcycle", 
    "bus", 
    "bicycle", 
    "auto rickshaw"
]


OBSTRUCTION_PHRASES: list[str] = [
    # 1. Universal Catch-Alls (Functional concepts)
    "obstacle",
    "obstruction",
    "road hazard",
    "blockage",
    "unidentified object",
    "foreign object",
    "large mass",

    # 2. Road Hazards, Debris & Materials
    "debris",
    "metal scrap",
    "boulder",
    "pothole",
    "fallen tree",
    "spilled load",
    "spilled cargo",
    "cardboard box",
    "tire tread",
    "barrier",
    "plastic barrier",
    "structure",

    # 3. Broad Non-Standard Transport & Equipment
    "aircraft",
    "watercraft",
    "rail vehicle",
    "heavy machinery",
    "industrial equipment",

    # 4. Living Entities & Road Markers
    "pedestrian",
    "animal",
    "traffic cone",

    # 5. Textiles, Covering Materials & Soft Cargo (including flexible containers)
    "fabric material",
    "loose textile",
    "soft cargo",
    "covering",
    "bag",
    "rock",
    "mirror",
    "cloth bundle",
    "bundled textiles",
    "flexible containment",

    # 6. Fallen or Abandoned Small Vehicles/Debris (NEW SECTION)
    "fallen bicycle",
    "lying bike",
    "abandoned bicycle",
    "downed cycle",
    "small stone",
    "large stone",
    "loose gravel",
    "pebbles on road",
    
    "tipped over motorcycle",
    "unattended scooter on road"
]

# Visualisation colour palette — distinct high-contrast colours (BGR)
COLOUR_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 128, 255),    # orange
    (128, 0, 255),    # purple
    (0, 255, 128),    # spring green
    (255, 128, 0),    # sky blue
    (128, 255, 0),    # chartreuse
    (0, 128, 128),    # olive
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def select_device() -> str:
    """Pick the best available compute device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask by *radius* pixels (morphological dilation)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def intersection_over_minimum(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection-over-Minimum (IoM) between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    minimum = min(mask_a.sum(), mask_b.sum())
    if minimum == 0:
        return 0.0
    return float(inter) / float(minimum)


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RoadObstacleSegmenter:
    def __init__(
        self,
        model_path: str = "sam3.pt",
        device: str | None = None,
        conf: float = 0.50,
        proximity_px: int = 15,
        iom_dedup_threshold: float = 0.7,
        show_labels: bool = False,
    ) -> None:
        self.device = device or select_device()
        self.conf = conf
        self.proximity_px = proximity_px
        self.iom_dedup_threshold = iom_dedup_threshold
        self.show_labels = show_labels

        print(f"🖥  Device: {self.device}")
        print(f"📦 Loading SAM 3 from '{model_path}' …")
        t0 = time.perf_counter()

        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = {
            "conf": self.conf,
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "device": self.device,
            "save": False,
            "verbose": False,
        }
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        print(f"   ✅ Model loaded in {time.perf_counter() - t0:.1f}s\n")

    def _segment_road(self) -> np.ndarray:
        self.predictor.reset_prompts()
        results = self.predictor(text=ROAD_PHRASES)
        result = results[0]

        h, w = result.orig_img.shape[:2]
        road_mask = np.zeros((h, w), dtype=bool)

        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()
            for i in range(masks_data.shape[0]):
                road_mask |= masks_data[i].astype(bool)

        coverage_pct = 100.0 * road_mask.sum() / road_mask.size
        print(f"   [Step 1] Road Coverage: {coverage_pct:.1f}%")
        if coverage_pct > 95.0:
            print("   ⚠️ Warning: Road covers >95% of image.")
        return road_mask

    def _segment_concepts(self, phrases: list[str], sweep_name: str) -> list[dict]:
        """Generic method to sweep for a specific category of objects."""
        self.predictor.reset_prompts()
        results = self.predictor(text=phrases)
        result = results[0]

        detections: list[dict] = []

        if result.masks is None or result.boxes is None:
            return detections

        masks_data = result.masks.data.cpu().numpy()
        boxes_data = result.boxes.data.cpu().numpy()
        names = result.names

        for i in range(masks_data.shape[0]):
            cls_id = int(boxes_data[i, 5])
            label = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else names[cls_id]
            score = float(boxes_data[i, 4])

            detections.append({
                "mask": masks_data[i].astype(bool),
                "label": label,
                "score": score,
                "bbox": boxes_data[i, :4],
                "category": sweep_name
            })

        return detections

    @staticmethod
    def _compute_road_boundary(road_mask: np.ndarray, proximity_px: int) -> np.ndarray:
        return dilate_mask(road_mask, proximity_px)

    def _filter_by_road_proximity(self, detections: list[dict], road_mask: np.ndarray) -> list[dict]:
        road_zone = self._compute_road_boundary(road_mask, self.proximity_px)
        kept: list[dict] = []

        for det in detections:
            obj_mask = det["mask"]
            touching = np.logical_and(obj_mask, road_zone).any()
            if not touching:
                continue

            overlap_pixels = np.logical_and(obj_mask, road_mask).sum()
            total_pixels = obj_mask.sum()
            road_overlap = float(overlap_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
            det["road_overlap"] = road_overlap
            kept.append(det)

        return kept

    def _suppress_vehicle_overlaps(self, obstructions: list[dict], vehicles: list[dict]) -> list[dict]:
        """Subtracts vehicles from obstructions to eliminate semantic overlap."""
        kept: list[dict] = []
        for obs in obstructions:
            suppressed = False
            for veh in vehicles:
                iom = intersection_over_minimum(obs["mask"], veh["mask"])
                if iom > self.iom_dedup_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(obs)
        return kept

    def _deduplicate(self, detections: list[dict]) -> list[dict]:
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        keep_flags = [True] * len(detections)

        for i in range(len(detections)):
            if not keep_flags[i]:
                continue
            for j in range(i + 1, len(detections)):
                if not keep_flags[j]:
                    continue
                iom = intersection_over_minimum(detections[i]["mask"], detections[j]["mask"])
                if iom > self.iom_dedup_threshold:
                    keep_flags[j] = False

        return [d for d, keep in zip(detections, keep_flags) if keep]

    def _visualise(
        self,
        image: np.ndarray,
        road_mask: np.ndarray,
        detections: list[dict],
        output_path: str,
        show: bool = False,
    ) -> np.ndarray:
        canvas = image.copy()

        # Road overlay (subtle golden-blue tint)
        road_overlay = canvas.copy()
        road_overlay[road_mask] = (
            road_overlay[road_mask].astype(np.float32) * 0.5
            + np.array([200, 150, 50], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        canvas = road_overlay

        # Road contour outline
        road_contours, _ = cv2.findContours(
            road_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, road_contours, -1, (255, 200, 50), 2)

        # Pure obstacle color overlays (no text/labels by default)
        for idx, det in enumerate(detections):
            colour = COLOUR_PALETTE[idx % len(COLOUR_PALETTE)]
            mask = det["mask"]

            overlay = canvas.copy()
            overlay[mask] = (
                overlay[mask].astype(np.float32) * 0.45
                + np.array(colour, dtype=np.float32) * 0.55
            ).astype(np.uint8)
            canvas = overlay

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, colour, 2)

            if self.show_labels:
                road_pct = det.get("road_overlap", 0.0)
                label_text = f"{det['label']} {det['score']:.0%} (road:{road_pct:.0%})"
                bbox = det["bbox"]
                x1, y1 = int(bbox[0]), int(bbox[1])
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                label_y = max(y1 - 8, th + 4)
                cv2.rectangle(canvas, (x1, label_y - th - 4), (x1 + tw + 6, label_y + baseline + 2), colour, -1)
                cv2.putText(
                    canvas, label_text,
                    (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, canvas)

        if show:
            fig, axes = plt.subplots(1, 3, figsize=(22, 7))
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image", fontsize=13)
            axes[0].axis("off")

            road_vis = image.copy()
            road_vis[road_mask] = (
                road_vis[road_mask].astype(np.float32) * 0.4
                + np.array([200, 150, 50], dtype=np.float32) * 0.6
            ).astype(np.uint8)
            axes[1].imshow(cv2.cvtColor(road_vis, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Step 1: Road Mask", fontsize=13)
            axes[1].axis("off")

            axes[2].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f"Final: {len(detections)} anomaly(s)", fontsize=13)
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

        return canvas

    def process_image(
        self,
        image_path: str,
        output_path: str,
        show: bool = False,
    ) -> list[dict]:
        """Runs segmentation on a single image and writes output."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"   ⚠️ Could not read image: {image_path}. Skipping.")
            return []

        # Resize to prevent VRAM/RAM explosion
        max_dim = 1024  # You can lower this to 800 or 640 if it still crashes
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        t_start = time.perf_counter()
        self.predictor.set_image(image)

        # 1. Road mask
        road_mask = self._segment_road()

        # 2. Traffic sweep
        vehicles = self._segment_concepts(VEHICLE_PHRASES, sweep_name="VEHICLES")

        # 3. Anomaly sweep
        obstructions = self._segment_concepts(OBSTRUCTION_PHRASES, sweep_name="OBSTRUCTIONS")

        # 4. Proximity filter (keep items on the road)
        vehicles = self._filter_by_road_proximity(vehicles, road_mask)
        obstructions = self._filter_by_road_proximity(obstructions, road_mask)

        # 5. Spatial subtraction (remove false anomalies that are vehicles)
        final_anomalies = self._suppress_vehicle_overlaps(obstructions, vehicles)

        # 6. Deduplicate remaining anomalies
        final_anomalies = self._deduplicate(final_anomalies)

        elapsed = time.perf_counter() - t_start

        # Visualise (clean colors only)
        self._visualise(image, road_mask, final_anomalies, output_path, show=show)

        self.predictor.reset_image()
        self.predictor.reset_prompts()

        print(f"   ⏱  Done in {elapsed:.2f}s | {len(final_anomalies)} obstacle(s) detected -> {output_path}")
        for det in final_anomalies:
            print(f"      • {det['label']:20s} score={det['score']:.3f} road_overlap={det.get('road_overlap', 0):.1%}")

        # Force garbage collection and clear GPU/MPS cache
        del image, road_mask, vehicles, obstructions
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return final_anomalies

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        show: bool = False,
    ) -> dict[str, list[dict]]:
        """Batch processes all images in input_dir and saves outputs to output_dir."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Gather image files
        image_files = sorted([
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        total = len(image_files)
        if total == 0:
            print(f"❌ No supported image files found in '{input_dir}'")
            return {}

        print(f"{'='*70}")
        print(f"  Batch Road Obstacle Segmentation")
        print(f"  Input Directory : {input_path} ({total} images)")
        print(f"  Output Directory: {output_path}")
        print(f"{'='*70}\n")

        all_results: dict[str, list[dict]] = {}
        t_batch_start = time.perf_counter()

        for idx, img_file in enumerate(image_files, 1):
            print(f"[{idx}/{total}] Processing {img_file.name} …")
            out_file = output_path / f"segmented_{img_file.stem}.jpg"

            try:
                detections = self.process_image(
                    image_path=str(img_file),
                    output_path=str(out_file),
                    show=show,
                )
                all_results[img_file.name] = detections
            except Exception as e:
                print(f"   ❌ Error processing {img_file.name}: {e}")

            print()

        total_elapsed = time.perf_counter() - t_batch_start
        total_obstacles = sum(len(dets) for dets in all_results.values())
        avg_time = total_elapsed / max(total, 1)

        print(f"{'='*70}")
        print(f"  BATCH INFERENCE COMPLETE")
        print(f"  Processed {len(all_results)}/{total} images successfully")
        print(f"  Total obstacles detected across dataset: {total_obstacles}")
        print(f"  Total time: {total_elapsed:.1f}s (avg {avg_time:.2f}s/image)")
        print(f"  Results saved to: {output_path.resolve()}")
        print(f"{'='*70}")

        return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM 3 Road Obstacle & Anomaly Segmentation Pipeline (Single & Batch Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="/Users/asadirfan358/Desktop/Empiristech/dir 3",
        help="Path to input image file or directory containing images (default: /Users/asadirfan358/Desktop/Empiristech/dir 3)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output_dir3",
        help="Path to output directory (for batch mode) or output file (for single image mode)",
    )
    parser.add_argument(
        "--model", "-m",
        default="sam3.pt",
        help="SAM 3 model weights path (default: sam3.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float, default=0.50,
        help="SAM 3 confidence threshold (default: 0.50)",
    )
    parser.add_argument(
        "--proximity",
        type=int, default=15,
        help="Road-touching proximity in pixels (default: 15)",
    )
    parser.add_argument(
        "--iom",
        type=float, default=0.7,
        help="IoM threshold for deduplication and vehicle subtraction (default: 0.7)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Draw text labels and bounding boxes (default: False, pure colors only)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pop up matplotlib visualization window for each image",
    )

    args = parser.parse_args()

    segmenter = RoadObstacleSegmenter(
        model_path=args.model,
        conf=args.conf,
        proximity_px=args.proximity,
        iom_dedup_threshold=args.iom,
        show_labels=args.show_labels,
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        # Batch directory mode
        segmenter.process_directory(
            input_dir=str(input_path),
            output_dir=args.output_dir,
            show=args.show,
        )
    elif input_path.is_file():
        # Single image mode
        out_file = args.output_dir
        if not out_file.lower().endswith(tuple(IMAGE_EXTENSIONS)):
            out_file = os.path.join(args.output_dir, f"segmented_{input_path.name}")
        segmenter.process_image(
            image_path=str(input_path),
            output_path=out_file,
            show=args.show,
        )
    else:
        print(f"❌ Error: Input path '{args.input}' does not exist.")


if __name__ == "__main__":
    main()