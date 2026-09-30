#!/usr/bin/env python3
"""
Road Obstacle Segmentation Pipeline using SAM 3
================================================

Strategy:
  1. Segment the road surface securely using a strict "road" prompt.
  2. Detect obstacles using targeted core concepts.
  3. Geometric proximity filter — keep only objects touching/overlapping the road.
  4. IoM-based deduplication for overlapping masks.
  5. Rich visualization with per-object labels and confidence scores.

Usage:
    python road_obstacle_segmentation.py                             
    python road_obstacle_segmentation.py --image /path/to/image.jpg  
"""

from __future__ import annotations

import argparse
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Keep it strict to prevent background bleeding
ROAD_PHRASES: list[str] = [
    "road"
]

# Targeted core concepts that worked previously without overloading the encoder
OBJECT_PHRASES: list[str] = [
    "buffalo",
    "animal",
    "vehicle",
    "obstacle",
    "debris",
    "person"
]

# Visualisation colour palette — distinct high-contrast colours (BGR)
COLOUR_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 255),      # red
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
]

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
        conf: float = 0.50, # Restored strict default confidence
        proximity_px: int = 15,
        iom_dedup_threshold: float = 0.7,
    ) -> None:
        
        self.device = device or select_device()
        self.conf = conf
        self.proximity_px = proximity_px
        self.iom_dedup_threshold = iom_dedup_threshold

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
        print(f"   ✅ Model loaded in {time.perf_counter() - t0:.1f}s")

    def _segment_road(self) -> np.ndarray:
        print("\n── Step 1: Segmenting the road surface ──")
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
        print(f"   Road coverage: {coverage_pct:.1f}% of image")
        if coverage_pct > 95.0:
            print("   ⚠️  Warning: Road covers almost entire image. Check confidence threshold.")
        return road_mask

    def _segment_objects(self) -> list[dict]:
        print("\n── Step 2: Sweeping candidate object phrases ──")
        self.predictor.reset_prompts()
        results = self.predictor(text=OBJECT_PHRASES)
        result = results[0]

        detections: list[dict] = []

        if result.masks is None or result.boxes is None:
            print("   No objects detected.")
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
            })

        print(f"   Found {len(detections)} detection(s)")
        for det in detections:
            print(f"      • {det['label']:25s}  score={det['score']:.3f}")

        return detections

    @staticmethod
    def _compute_road_boundary(road_mask: np.ndarray, proximity_px: int) -> np.ndarray:
        return dilate_mask(road_mask, proximity_px)

    def _filter_by_road_proximity(self, detections: list[dict], road_mask: np.ndarray) -> list[dict]:
        print(f"\n── Step 3: Geometric proximity filter (radius={self.proximity_px}px) ──")
        road_zone = self._compute_road_boundary(road_mask, self.proximity_px)

        kept: list[dict] = []
        for det in detections:
            obj_mask = det["mask"]
            touching = np.logical_and(obj_mask, road_zone).any()
            if not touching:
                print(f"   ✗ {det['label']:25s}  — NOT touching the road → removed")
                continue

            overlap_pixels = np.logical_and(obj_mask, road_mask).sum()
            total_pixels = obj_mask.sum()
            road_overlap = float(overlap_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
            det["road_overlap"] = road_overlap

            print(f"   ✓ {det['label']:25s}  road_overlap={road_overlap:.1%}")
            kept.append(det)

        print(f"   Kept {len(kept)} / {len(detections)} detections")
        return kept

    def _deduplicate(self, detections: list[dict]) -> list[dict]:
        print(f"\n── Step 4: IoM deduplication (threshold={self.iom_dedup_threshold}) ──")
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
                    print(
                        f"   ⊘ '{detections[j]['label']}' (score={detections[j]['score']:.3f}) "
                        f"suppressed by '{detections[i]['label']}' (IoM={iom:.2f})"
                    )
                    keep_flags[j] = False

        result = [d for d, keep in zip(detections, keep_flags) if keep]
        removed = len(detections) - len(result)
        if removed:
            print(f"   Removed {removed} duplicate(s), {len(result)} unique objects remain")
        else:
            print("   No duplicates found")
        return result

    @staticmethod
    def _visualise(
        image: np.ndarray,
        road_mask: np.ndarray,
        detections: list[dict],
        output_path: str,
        show: bool = True,
    ) -> np.ndarray:
        print(f"\n── Step 5: Visualisation ──")
        canvas = image.copy()

        road_overlay = canvas.copy()
        road_overlay[road_mask] = (
            road_overlay[road_mask].astype(np.float32) * 0.5
            + np.array([200, 150, 50], dtype=np.float32) * 0.5
        ).astype(np.uint8)
        canvas = road_overlay

        road_contours, _ = cv2.findContours(
            road_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, road_contours, -1, (255, 200, 50), 2)

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


        cv2.imwrite(output_path, canvas)
        print(f"   💾 Saved → {output_path}")

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
            axes[2].set_title(f"Final: {len(detections)} object(s) on road", fontsize=13)
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()

        return canvas

    def run(
        self,
        image_path: str,
        output_path: str = "road_obstacles_output.jpg",
        show: bool = True,
    ) -> list[dict]:
        print(f"\n{'='*60}")
        print(f"  Road Obstacle Segmentation Pipeline")
        print(f"  Image: {image_path}")
        print(f"{'='*60}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        t_start = time.perf_counter()
        self.predictor.set_image(image_path)

        road_mask = self._segment_road()
        
        self.predictor.reset_prompts()
        detections = self._segment_objects()

        detections = self._filter_by_road_proximity(detections, road_mask)
        detections = self._deduplicate(detections)

        elapsed = time.perf_counter() - t_start
        print(f"\n⏱  Pipeline completed in {elapsed:.1f}s")

        self._visualise(image, road_mask, detections, output_path, show=show)

        self.predictor.reset_image()
        self.predictor.reset_prompts()

        return detections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM 3 Road Obstacle Segmentation Pipeline",
    )
    parser.add_argument(
        "--image", "-i",
        default="/Users/asadirfan358/Desktop/Empiristech/images.jpeg",
        help="Path to input image",
    )
    parser.add_argument(
        "--output", "-o",
        default="road_obstacles_output.jpg",
        help="Output image path",
    )
    parser.add_argument(
        "--model", "-m",
        default="sam3.pt",
        help="SAM 3 model weights path",
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
        help="IoM threshold for deduplication (default: 0.7)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip matplotlib display (headless mode)",
    )

    args = parser.parse_args()

    segmenter = RoadObstacleSegmenter(
        model_path=args.model,
        conf=args.conf,
        proximity_px=args.proximity,
        iom_dedup_threshold=args.iom,
    )

    segmenter.run(
        image_path=args.image,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()