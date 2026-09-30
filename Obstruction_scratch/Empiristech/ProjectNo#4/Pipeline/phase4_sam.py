"""
phase4_sam.py — Precision Area Segmentation with SAM 3

Phase 4 of the Hazard Zone Segmentation Pipeline.

Purpose:
    Takes the consolidated point prompts (coordinates + labels) from Phase 3
    and feeds them to the SAM 3 model via the REST API at localhost:8001.

    SAM 3 uses the positive points (label=1) as seeds to flood-fill the open
    floor area of the hazard zone, while negative points (label=0) enforce
    hard stop boundaries at the painted floor lines, equipment bases, and
    zone perimeters.

    The resulting binary mask is overlaid onto the original image as a
    semi-transparent colored highlight for visual verification.

Outputs:
    segmentation_mask.png — raw binary mask from SAM 3
    final_overlay.jpg     — original image with mask overlay

Usage (standalone):
    python phase4_sam.py --image /path/to/original.png \\
                         --prompts /path/to/sam_prompts.json
"""

import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests

# ── Import pipeline configuration ───────────────────────────
try:
    from config import (
        OUTPUT_DIR,
        SAM3_API_URL,
        SAM3_MAX_RETRIES,
        SAM3_RETRY_DELAY,
        OVERLAY_COLOR_BGR,
        OVERLAY_ALPHA,
        SEGMENTATION_MASK_FILENAME,
        FINAL_OVERLAY_FILENAME,
    )
except ImportError:
    from Pipeline.config import (
        OUTPUT_DIR,
        SAM3_API_URL,
        SAM3_MAX_RETRIES,
        SAM3_RETRY_DELAY,
        OVERLAY_COLOR_BGR,
        OVERLAY_ALPHA,
        SEGMENTATION_MASK_FILENAME,
        FINAL_OVERLAY_FILENAME,
    )


# ══════════════════════════════════════════════════════════════
#  Helper: Decode a binary mask from SAM 3 response
# ══════════════════════════════════════════════════════════════
def _decode_mask(response: requests.Response, image_shape: tuple) -> np.ndarray | None:
    """
    Attempt to extract a grayscale mask from the SAM 3 API response.

    Handles multiple response formats:
      1. Raw image bytes (Content-Type: image/*)
      2. JSON with detections[].mask_b64 (Ultralytics-style)
      3. JSON with mask_b64 at top level

    Returns a single-channel grayscale mask or None on failure.
    """
    content_type = response.headers.get("Content-Type", "")

    # ── Format 1: Direct image response ─────────────────────
    if "image" in content_type:
        mask_bytes = response.content
        mask_array = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            print("[Phase 4] Decoded mask from raw image response")
            return mask

    # ── Format 2/3: JSON response ───────────────────────────
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        return None

    b64_mask = None
    score = 0.0

    # Ultralytics-style: detections array
    if "detections" in data and isinstance(data["detections"], list):
        if len(data["detections"]) > 0:
            # Pick the highest-scoring detection
            best = max(data["detections"], key=lambda d: d.get("score", 0))
            b64_mask = best.get("mask_b64")
            score = best.get("score", 0)

    # Top-level mask_b64
    elif "mask_b64" in data:
        b64_mask = data["mask_b64"]

    # Top-level mask as nested list (some SAM APIs return the mask as a 2D array)
    elif "mask" in data and isinstance(data["mask"], list):
        try:
            mask = np.array(data["mask"], dtype=np.uint8)
            if mask.ndim == 2:
                print(f"[Phase 4] Decoded mask from JSON array ({mask.shape})")
                return mask
        except Exception:
            pass

    if b64_mask:
        # Strip optional data URI prefix
        if "," in b64_mask:
            b64_mask = b64_mask.split(",")[1]
        mask_bytes = base64.b64decode(b64_mask)
        mask_array = np.frombuffer(mask_bytes, np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            print(f"[Phase 4] Decoded base64 mask (score: {score:.3f})")
            return mask

    # ── Fallback: save raw JSON for debugging ───────────────
    debug_path = os.path.join(OUTPUT_DIR, "sam3_raw_response.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(debug_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Phase 4] ⚠️  Unknown JSON format. Saved raw response → {debug_path}")
    return None


# ══════════════════════════════════════════════════════════════
#  Core Function: Segment with SAM 3
# ══════════════════════════════════════════════════════════════
def segment_with_sam(
    image_path: str,
    points: np.ndarray,
    labels: np.ndarray,
    save_outputs: bool = True,
) -> np.ndarray | None:
    """
    Send the original image and point prompts to the SAM 3 API.

    Parameters
    ----------
    image_path : str
        Path to the original (non-gridded) image.
    points : np.ndarray
        Shape (N, 2) array of [x, y] pixel coordinates.
    labels : np.ndarray
        Shape (N,) array of 0 (negative) or 1 (positive) labels.
    save_outputs : bool
        If True, saves mask and overlay to OUTPUT_DIR.

    Returns
    -------
    np.ndarray or None
        The binary segmentation mask, or None if all attempts failed.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise IOError(f"Failed to load image: {image_path}")

    img_h, img_w = original_img.shape[:2]
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    print(f"[Phase 4] Image: {img_w}×{img_h} px")
    print(f"[Phase 4] Point prompts: {len(points)} total ({n_positive} positive, {n_negative} negative)")

    # ── Prepare the API payload ─────────────────────────────
    # Convert numpy arrays to JSON-serializable lists
    points_json = points.tolist()
    labels_json = labels.tolist()

    mask = None

    for attempt in range(SAM3_MAX_RETRIES):
        print(f"\n[Phase 4] ── SAM 3 Attempt {attempt + 1}/{SAM3_MAX_RETRIES} ──")
        try:
            with open(image_path, "rb") as img_file:
                files = {"file": (os.path.basename(image_path), img_file, "image/png")}
                payload = {
                    "points": json.dumps(points_json),
                    "point_labels": json.dumps(labels_json), # Renamed to avoid 'text' clash
                }

                response = requests.post(
                    SAM3_API_URL,
                    files=files,
                    data=payload,
                    timeout=120,
                )

            if response.status_code == 200:
                mask = _decode_mask(response, (img_h, img_w))
                if mask is not None:
                    print(f"[Phase 4] ✅ SAM 3 returned a valid mask ({mask.shape})")
                    break
                else:
                    print("[Phase 4] ⚠️  200 OK but could not decode mask. Retrying...")
            else:
                print(f"[Phase 4] ❌ SAM 3 returned HTTP {response.status_code}")
                try:
                    err_body = response.text[:500]
                    print(f"           Response: {err_body}")
                except Exception:
                    pass

        except requests.exceptions.ConnectionError:
            print(f"[Phase 4] ❌ Could not connect to SAM 3 at {SAM3_API_URL}")
        except requests.exceptions.Timeout:
            print("[Phase 4] ❌ SAM 3 request timed out (120s)")
        except Exception as e:
            print(f"[Phase 4] ❌ Unexpected error: {e}")

        if attempt < SAM3_MAX_RETRIES - 1:
            print(f"           Retrying in {SAM3_RETRY_DELAY}s...")
            time.sleep(SAM3_RETRY_DELAY)

    if mask is None:
        print(f"\n[Phase 4] ⚠️  All {SAM3_MAX_RETRIES} attempts failed. No mask produced.")
        return None

    # ── Ensure mask dimensions match the original image ─────
    if mask.shape[:2] != (img_h, img_w):
        print(f"[Phase 4] Resizing mask from {mask.shape[:2]} to ({img_h}, {img_w})")
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    # ── Threshold to clean binary mask ──────────────────────
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # ── Compute mask statistics ─────────────────────────────
    mask_pixels = int(np.count_nonzero(binary_mask))
    total_pixels = img_h * img_w
    coverage_pct = (mask_pixels / total_pixels) * 100
    print(f"[Phase 4] Mask coverage: {mask_pixels:,} px / {total_pixels:,} px ({coverage_pct:.1f}%)")

    # ── Create colored overlay ──────────────────────────────
    color_mask = np.zeros_like(original_img)
    color_mask[binary_mask == 255] = OVERLAY_COLOR_BGR

    overlay = cv2.addWeighted(original_img, 1.0, color_mask, OVERLAY_ALPHA, 0)

    # Draw a thin contour around the mask edge for crisp definition
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, OVERLAY_COLOR_BGR, 2)

    print(f"[Phase 4] Created overlay with {len(contours)} contour region(s)")

    # ── Save outputs ────────────────────────────────────────
    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        mask_path = os.path.join(OUTPUT_DIR, SEGMENTATION_MASK_FILENAME)
        overlay_path = os.path.join(OUTPUT_DIR, FINAL_OVERLAY_FILENAME)

        cv2.imwrite(mask_path, binary_mask)
        print(f"[Phase 4] Saved binary mask → {mask_path}")

        cv2.imwrite(overlay_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[Phase 4] Saved final overlay → {overlay_path}")

    return binary_mask


# ══════════════════════════════════════════════════════════════
#  Standalone Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4 — SAM 3 point-prompt segmentation with overlay"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to original image")
    parser.add_argument("--prompts", "-p", required=True, help="Path to sam_prompts.json")

    args = parser.parse_args()

    try:
        # Load prompts from the Phase 3 debug file
        with open(args.prompts, "r") as f:
            prompt_data = json.load(f)

        pts = []
        lbls = []
        for record in prompt_data.get("points", []):
            pts.append([record["x"], record["y"]])
            lbls.append(record["label"])

        points_arr = np.array(pts, dtype=np.float32)
        labels_arr = np.array(lbls, dtype=np.int32)

        result = segment_with_sam(args.image, points_arr, labels_arr)
        if result is not None:
            print(f"\n✅ Phase 4 complete. Mask shape: {result.shape}")
        else:
            print("\n⚠️  Phase 4 finished without producing a mask.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Phase 4 failed: {e}")
        sys.exit(1)
