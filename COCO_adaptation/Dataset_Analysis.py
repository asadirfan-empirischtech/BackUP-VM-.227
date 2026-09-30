"""
=============================================================================
COCO Dataset Analyzer — Exhaustive Annotation & Image Statistics
=============================================================================
Reads a standard COCO-format dataset (instances, captions, keypoints) and
prints a highly detailed report AND saves the full analysis to a JSON file.

Usage:
    python 1.py                          # uses DATASET_ROOT below
    python 1.py /path/to/coco_dataset    # override via CLI argument

Output:
    dataset_analysis.json  (saved next to this script or in DATASET_ROOT)
=============================================================================
"""

import json
import os
import sys
import math
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION — set the root of your COCO dataset here
# ============================================================================
DATASET_ROOT = r"/home/azureuser/Documents/COCO_adaptation/"  # <-- CHANGE THIS

# Override from CLI if provided
if len(sys.argv) > 1:
    DATASET_ROOT = sys.argv[1]

# Where to save the JSON report
JSON_OUTPUT_PATH = os.path.join(DATASET_ROOT, "dataset_analysis.json")

# ============================================================================
# HELPERS
# ============================================================================

def load_json(path):
    """Load a JSON annotation file and return the parsed dict."""
    print(f"  Loading {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.1f} MB) ...")
    with open(path, "r") as f:
        return json.load(f)


def fmt(n):
    return f"{n:,}"


def pct(part, total):
    return f"{100.0 * part / total:.2f}%" if total else "N/A"


def pct_val(part, total):
    """Return the numeric percentage value (for JSON)."""
    return round(100.0 * part / total, 4) if total else None


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def stats_summary(values, label="value"):
    """Return a dict of descriptive statistics for a list of numbers."""
    if not values:
        return {f"{label}_count": 0}
    s = sorted(values)
    n = len(s)
    total = sum(s)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)
    return {
        f"{label}_count": n,
        f"{label}_sum": round(total, 4),
        f"{label}_min": round(s[0], 4),
        f"{label}_max": round(s[-1], 4),
        f"{label}_mean": round(mean, 4),
        f"{label}_median": round(percentile(s, 50), 4),
        f"{label}_std": round(std, 4),
        f"{label}_p5": round(percentile(s, 5), 4),
        f"{label}_p25": round(percentile(s, 25), 4),
        f"{label}_p75": round(percentile(s, 75), 4),
        f"{label}_p95": round(percentile(s, 95), 4),
    }


def histogram_buckets(values, n_buckets=10):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [(f"{lo:.1f}", len(values))]
    step = (hi - lo) / n_buckets
    buckets = [0] * n_buckets
    for v in values:
        idx = min(int((v - lo) / step), n_buckets - 1)
        buckets[idx] += 1
    labels = []
    for i in range(n_buckets):
        edge_lo = lo + i * step
        edge_hi = lo + (i + 1) * step
        labels.append({
            "range": f"{edge_lo:.1f}–{edge_hi:.1f}",
            "range_low": round(edge_lo, 2),
            "range_high": round(edge_hi, 2),
            "count": buckets[i],
        })
    return labels


def print_histogram(buckets, title, bar_width=40):
    if not buckets:
        return
    max_count = max(b["count"] for b in buckets)
    print(f"\n  {title}")
    print(f"  {'Bucket':>20s}  {'Count':>8s}  Bar")
    print(f"  {'─' * 20}  {'─' * 8}  {'─' * bar_width}")
    for b in buckets:
        bar_len = int(bar_width * b["count"] / max_count) if max_count else 0
        print(f"  {b['range']:>20s}  {b['count']:>8,}  {'█' * bar_len}")


def section(title):
    width = 78
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def subsection(title):
    print(f"\n  ── {title} {'─' * max(0, 70 - len(title))}")


# ============================================================================
# ANNOTATION FILE DISCOVERY
# ============================================================================

def discover_files(root):
    ann_dir = os.path.join(root, "annotations")
    if not os.path.isdir(ann_dir):
        print(f"ERROR: annotations/ directory not found under {root}")
        sys.exit(1)

    files = {}
    expected = {
        "instances_train": "instances_train2017.json",
        "instances_val": "instances_val2017.json",
        "captions_train": "captions_train2017.json",
        "captions_val": "captions_val2017.json",
        "keypoints_train": "person_keypoints_train2017.json",
        "keypoints_val": "person_keypoints_val2017.json",
    }
    for key, fname in expected.items():
        path = os.path.join(ann_dir, fname)
        files[key] = path if os.path.isfile(path) else None

    img_dirs = {}
    for split_name in ("train2017", "val2017"):
        d = os.path.join(root, split_name)
        img_dirs[split_name] = d if os.path.isdir(d) else None

    return files, img_dirs


# ============================================================================
# INSTANCE ANALYSIS
# ============================================================================

def analyze_instances(data, split_name):
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_id_to_super = {c["id"]: c.get("supercategory", "unknown") for c in categories}

    subsection(f"Instances — {split_name}")
    print(f"    Images          : {fmt(len(images))}")
    print(f"    Annotations     : {fmt(len(annotations))}")
    print(f"    Categories      : {fmt(len(categories))}")

    # Supercategories
    supercats_map = defaultdict(list)
    for cid, sc in cat_id_to_super.items():
        supercats_map[sc].append(cat_id_to_name[cid])
    supercats_json = {sc: sorted(members) for sc, members in sorted(supercats_map.items())}

    print(f"    Supercategories : {fmt(len(supercats_json))}")
    for sc, members in supercats_json.items():
        print(f"      • {sc:20s} ({len(members):>2d}): {', '.join(members)}")

    # --- Per-category counts ---
    cat_counts = Counter()
    cat_areas = defaultdict(list)
    cat_crowd = Counter()
    cat_bbox_aspects = defaultdict(list)
    objs_per_image = Counter()

    SIZE_SMALL = 32 ** 2
    SIZE_MEDIUM = 96 ** 2
    size_dist = {"small": 0, "medium": 0, "large": 0}

    for ann in annotations:
        cid = ann["category_id"]
        cat_counts[cid] += 1
        area = ann.get("area", 0)
        cat_areas[cid].append(area)
        objs_per_image[ann["image_id"]] += 1
        if ann.get("iscrowd", 0):
            cat_crowd[cid] += 1
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            _, _, bw, bh = bbox
            if bh > 0:
                cat_bbox_aspects[cid].append(bw / bh)
        if area < SIZE_SMALL:
            size_dist["small"] += 1
        elif area < SIZE_MEDIUM:
            size_dist["medium"] += 1
        else:
            size_dist["large"] += 1

    total_anns = len(annotations)

    # Print per-category table
    subsection("Per-Category Object Counts (sorted by frequency)")
    print(f"    {'#':>4s}  {'Category':25s}  {'Count':>10s}  {'%':>7s}  {'Crowd':>7s}  {'AvgArea':>10s}  {'AvgAspect':>10s}")
    print(f"    {'─' * 4}  {'─' * 25}  {'─' * 10}  {'─' * 7}  {'─' * 7}  {'─' * 10}  {'─' * 10}")

    per_category_json = []
    for rank, (cid, count) in enumerate(cat_counts.most_common(), 1):
        name = cat_id_to_name.get(cid, f"id={cid}")
        crowd = cat_crowd.get(cid, 0)
        avg_area = sum(cat_areas[cid]) / len(cat_areas[cid]) if cat_areas[cid] else 0
        aspects = cat_bbox_aspects.get(cid, [])
        avg_aspect = sum(aspects) / len(aspects) if aspects else 0
        print(f"    {rank:4d}  {name:25s}  {count:>10,}  {pct(count, total_anns):>7s}"
              f"  {crowd:>7,}  {avg_area:>10,.0f}  {avg_aspect:>10.3f}")
        per_category_json.append({
            "rank": rank,
            "category_id": cid,
            "category_name": name,
            "supercategory": cat_id_to_super.get(cid, "unknown"),
            "count": count,
            "percentage": pct_val(count, total_anns),
            "crowd_count": crowd,
            "avg_area_px2": round(avg_area, 2),
            "avg_bbox_aspect_ratio": round(avg_aspect, 4),
        })

    # --- Size distribution ---
    subsection("Object Size Distribution (COCO thresholds)")
    print(f"    Small  (area < {SIZE_SMALL:>6,} px²) : {fmt(size_dist['small']):>10s}  ({pct(size_dist['small'], total_anns)})")
    print(f"    Medium (area < {SIZE_MEDIUM:>6,} px²) : {fmt(size_dist['medium']):>10s}  ({pct(size_dist['medium'], total_anns)})")
    print(f"    Large  (area ≥ {SIZE_MEDIUM:>6,} px²) : {fmt(size_dist['large']):>10s}  ({pct(size_dist['large'], total_anns)})")

    size_dist_json = {
        "small": {"threshold": f"area < {SIZE_SMALL} px²", "count": size_dist["small"], "percentage": pct_val(size_dist["small"], total_anns)},
        "medium": {"threshold": f"area < {SIZE_MEDIUM} px²", "count": size_dist["medium"], "percentage": pct_val(size_dist["medium"], total_anns)},
        "large": {"threshold": f"area >= {SIZE_MEDIUM} px²", "count": size_dist["large"], "percentage": pct_val(size_dist["large"], total_anns)},
    }

    # --- Objects per image ---
    img_ids_with_anns = set(objs_per_image.keys())
    for img in images:
        if img["id"] not in img_ids_with_anns:
            objs_per_image[img["id"]] = 0

    counts_list = list(objs_per_image.values())
    opi_stats = stats_summary(counts_list, "objs_per_img")
    subsection("Objects per Image")
    for k, v in opi_stats.items():
        print(f"    {k:30s}: {v}")
    imgs_with_zero = sum(1 for c in counts_list if c == 0)
    print(f"    images_with_0_annotations  : {fmt(imgs_with_zero)} ({pct(imgs_with_zero, len(images))})")

    opi_histogram = histogram_buckets(counts_list, 15)
    print_histogram(opi_histogram, "Objects-per-Image Histogram")
    opi_stats["images_with_0_annotations"] = imgs_with_zero

    # --- Bbox area distribution ---
    all_areas = [a for areas in cat_areas.values() for a in areas]
    area_stats = {}
    if all_areas:
        area_stats = stats_summary(all_areas, "bbox_area")
        subsection("Bounding Box Area (px²)")
        for k, v in area_stats.items():
            print(f"    {k:30s}: {v:,.2f}" if isinstance(v, float) else f"    {k:30s}: {v}")

    # --- Crowd annotations ---
    total_crowd = sum(cat_crowd.values())
    subsection("Crowd Annotations (iscrowd=1)")
    print(f"    Total crowd annotations : {fmt(total_crowd)} ({pct(total_crowd, total_anns)})")

    # --- Category co-occurrence ---
    img_to_cats = defaultdict(set)
    for ann in annotations:
        img_to_cats[ann["image_id"]].add(ann["category_id"])
    pair_counts = Counter()
    for cats in img_to_cats.values():
        cats_sorted = sorted(cats)
        for i in range(len(cats_sorted)):
            for j in range(i + 1, len(cats_sorted)):
                pair_counts[(cats_sorted[i], cats_sorted[j])] += 1

    subsection("Top 20 Category Co-occurrence Pairs (in same image)")
    print(f"    {'Category A':25s}  {'Category B':25s}  {'Images':>8s}")
    print(f"    {'─' * 25}  {'─' * 25}  {'─' * 8}")
    cooccurrence_json = []
    for (ca, cb), cnt in pair_counts.most_common(20):
        na = cat_id_to_name.get(ca, str(ca))
        nb = cat_id_to_name.get(cb, str(cb))
        print(f"    {na:25s}  {nb:25s}  {cnt:>8,}")
        cooccurrence_json.append({"category_a": na, "category_b": nb, "images": cnt})

    return {
        "images": len(images),
        "annotations": total_anns,
        "categories": len(categories),
        "supercategories": supercats_json,
        "per_category": per_category_json,
        "size_distribution": size_dist_json,
        "objects_per_image": opi_stats,
        "objects_per_image_histogram": opi_histogram,
        "bbox_area_stats": area_stats,
        "crowd_annotations": {"total": total_crowd, "percentage": pct_val(total_crowd, total_anns)},
        "top_20_cooccurrence": cooccurrence_json,
        "cat_counts": cat_counts,
        "cat_id_to_name": cat_id_to_name,
        "size_dist": size_dist,
        "crowd": total_crowd,
    }


# ============================================================================
# IMAGE ANALYSIS
# ============================================================================

def analyze_images(data, split_name, img_dir):
    images = data.get("images", [])
    if not images:
        return {}

    subsection(f"Image Metadata — {split_name}")

    widths = [img["width"] for img in images]
    heights = [img["height"] for img in images]
    aspects = [w / h for w, h in zip(widths, heights) if h > 0]

    res_counter = Counter()
    for w, h in zip(widths, heights):
        res_counter[(w, h)] += 1

    print(f"    Total images: {fmt(len(images))}")
    print(f"    Unique resolutions: {fmt(len(res_counter))}")
    print(f"\n    {'Resolution':>15s}  {'Count':>8s}  {'%':>7s}")
    print(f"    {'─' * 15}  {'─' * 8}  {'─' * 7}")

    resolution_json = []
    for (w, h), cnt in res_counter.most_common(15):
        print(f"    {w}×{h:>5d}  {cnt:>8,}  {pct(cnt, len(images)):>7s}")
        resolution_json.append({"width": w, "height": h, "count": cnt, "percentage": pct_val(cnt, len(images))})

    aspect_stats = stats_summary(aspects, "aspect_ratio")
    subsection(f"Aspect Ratios — {split_name}")
    for k, v in aspect_stats.items():
        print(f"    {k:30s}: {v}")

    file_check = {}
    if img_dir and os.path.isdir(img_dir):
        actual_files = set(os.listdir(img_dir))
        expected_files = {img["file_name"] for img in images}
        missing = expected_files - actual_files
        extra = actual_files - expected_files
        print(f"\n    Image directory     : {img_dir}")
        print(f"    Files on disk      : {fmt(len(actual_files))}")
        print(f"    Expected from JSON : {fmt(len(expected_files))}")
        print(f"    Missing files      : {fmt(len(missing))}")
        print(f"    Extra files        : {fmt(len(extra))}")
        file_check = {
            "directory": img_dir,
            "files_on_disk": len(actual_files),
            "expected_from_json": len(expected_files),
            "missing_files": len(missing),
            "extra_files": len(extra),
            "missing_filenames": sorted(missing)[:50] if missing else [],
        }
    else:
        print(f"\n    Image directory not found: {img_dir} (skipping file check)")
        file_check = {"directory": img_dir, "status": "not_found"}

    return {
        "total_images": len(images),
        "unique_resolutions": len(res_counter),
        "top_15_resolutions": resolution_json,
        "aspect_ratio_stats": aspect_stats,
        "file_check": file_check,
    }


# ============================================================================
# SEGMENTATION ANALYSIS
# ============================================================================

def analyze_segmentation(data, split_name):
    annotations = data.get("annotations", [])
    if not annotations:
        return {}

    subsection(f"Segmentation Masks — {split_name}")

    polygon_count = 0
    rle_count = 0
    no_seg_count = 0
    polygon_vertex_counts = []

    for ann in annotations:
        seg = ann.get("segmentation")
        if seg is None:
            no_seg_count += 1
        elif isinstance(seg, list):
            polygon_count += 1
            total_vertices = sum(len(poly) // 2 for poly in seg if isinstance(poly, list))
            polygon_vertex_counts.append(total_vertices)
        elif isinstance(seg, dict):
            rle_count += 1
        else:
            no_seg_count += 1

    total = len(annotations)
    print(f"    Polygon masks      : {fmt(polygon_count):>10s}  ({pct(polygon_count, total)})")
    print(f"    RLE masks (crowd)  : {fmt(rle_count):>10s}  ({pct(rle_count, total)})")
    print(f"    No segmentation    : {fmt(no_seg_count):>10s}  ({pct(no_seg_count, total)})")

    vertex_stats = {}
    if polygon_vertex_counts:
        vertex_stats = stats_summary(polygon_vertex_counts, "vertices_per_mask")
        subsection(f"Polygon Vertices per Mask — {split_name}")
        for k, v in vertex_stats.items():
            print(f"    {k:30s}: {v}")

    return {
        "polygon_masks": {"count": polygon_count, "percentage": pct_val(polygon_count, total)},
        "rle_masks_crowd": {"count": rle_count, "percentage": pct_val(rle_count, total)},
        "no_segmentation": {"count": no_seg_count, "percentage": pct_val(no_seg_count, total)},
        "polygon_vertex_stats": vertex_stats,
    }


# ============================================================================
# CAPTION ANALYSIS
# ============================================================================

def analyze_captions(data, split_name):
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not annotations:
        return {}

    subsection(f"Captions — {split_name}")
    print(f"    Images   : {fmt(len(images))}")
    print(f"    Captions : {fmt(len(annotations))}")
    captions_per_img_avg = len(annotations) / len(images) if images else 0
    print(f"    Captions/image (avg) : {captions_per_img_avg:.2f}" if images else "")

    caption_lengths_chars = []
    caption_lengths_words = []
    all_words = []
    captions_per_image = Counter()

    for ann in annotations:
        caption = ann.get("caption", "")
        words = caption.lower().split()
        caption_lengths_chars.append(len(caption))
        caption_lengths_words.append(len(words))
        all_words.extend(words)
        captions_per_image[ann["image_id"]] += 1

    vocab = set(all_words)
    word_freq = Counter(all_words)

    print(f"    Total words        : {fmt(len(all_words))}")
    print(f"    Unique vocabulary  : {fmt(len(vocab))}")

    words_stats = stats_summary(caption_lengths_words, "words_per_caption")
    subsection(f"Caption Length (words) — {split_name}")
    for k, v in words_stats.items():
        print(f"    {k:30s}: {v}")

    chars_stats = stats_summary(caption_lengths_chars, "chars_per_caption")
    subsection(f"Caption Length (characters) — {split_name}")
    for k, v in chars_stats.items():
        print(f"    {k:30s}: {v}")

    subsection(f"Captions per Image — {split_name}")
    caps_list = list(captions_per_image.values())
    cap_dist = Counter(caps_list)
    captions_per_image_dist = {}
    for n_caps, n_imgs in sorted(cap_dist.items()):
        print(f"    {n_caps} captions : {fmt(n_imgs)} images")
        captions_per_image_dist[str(n_caps)] = n_imgs

    subsection(f"Top 50 Most Frequent Words — {split_name}")
    print(f"    {'Rank':>5s}  {'Word':20s}  {'Count':>10s}  {'%':>7s}")
    print(f"    {'─' * 5}  {'─' * 20}  {'─' * 10}  {'─' * 7}")
    top_words_json = []
    for rank, (word, cnt) in enumerate(word_freq.most_common(50), 1):
        print(f"    {rank:5d}  {word:20s}  {cnt:>10,}  {pct(cnt, len(all_words)):>7s}")
        top_words_json.append({"rank": rank, "word": word, "count": cnt, "percentage": pct_val(cnt, len(all_words))})

    words_histogram = histogram_buckets(caption_lengths_words, 12)
    print_histogram(words_histogram, f"Words-per-Caption Histogram — {split_name}")

    return {
        "images": len(images),
        "captions": len(annotations),
        "captions_per_image_avg": round(captions_per_img_avg, 2),
        "total_words": len(all_words),
        "unique_vocabulary": len(vocab),
        "words_per_caption_stats": words_stats,
        "chars_per_caption_stats": chars_stats,
        "captions_per_image_distribution": captions_per_image_dist,
        "top_50_words": top_words_json,
        "words_per_caption_histogram": words_histogram,
    }


# ============================================================================
# KEYPOINT ANALYSIS
# ============================================================================

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def analyze_keypoints(data, split_name):
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    if not annotations:
        return {}

    kp_names = COCO_KEYPOINT_NAMES
    for cat in categories:
        if cat.get("keypoints"):
            kp_names = cat["keypoints"]
            break

    n_kps = len(kp_names)

    subsection(f"Person Keypoints — {split_name}")
    print(f"    Images       : {fmt(len(images))}")
    print(f"    Annotations  : {fmt(len(annotations))}")
    print(f"    Keypoint names ({n_kps}): {', '.join(kp_names)}")

    vis_counts = {name: {0: 0, 1: 0, 2: 0} for name in kp_names}
    num_visible_per_ann = []
    num_labeled_per_ann = []
    complete_persons = 0

    for ann in annotations:
        kps = ann.get("keypoints", [])
        if not kps:
            continue
        n_vis = 0
        n_lab = 0
        for i in range(n_kps):
            if i * 3 + 2 < len(kps):
                v = int(kps[i * 3 + 2])
                vis_counts[kp_names[i]][v] += 1
                if v > 0:
                    n_lab += 1
                if v == 2:
                    n_vis += 1
        num_visible_per_ann.append(n_vis)
        num_labeled_per_ann.append(n_lab)
        if n_vis == n_kps:
            complete_persons += 1

    print(f"    Fully visible persons (all {n_kps} kps visible): {fmt(complete_persons)}"
          f" ({pct(complete_persons, len(annotations))})")

    subsection(f"Keypoint Visibility Breakdown — {split_name}")
    print(f"    {'Keypoint':20s}  {'NotLabeled':>10s}  {'Occluded':>10s}  {'Visible':>10s}  {'% Visible':>10s}")
    print(f"    {'─' * 20}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 10}")
    kp_visibility_json = []
    for name in kp_names:
        v0, v1, v2 = vis_counts[name][0], vis_counts[name][1], vis_counts[name][2]
        total = v0 + v1 + v2
        print(f"    {name:20s}  {v0:>10,}  {v1:>10,}  {v2:>10,}  {pct(v2, total):>10s}")
        kp_visibility_json.append({
            "keypoint": name,
            "not_labeled": v0,
            "occluded": v1,
            "visible": v2,
            "visible_percentage": pct_val(v2, total),
        })

    vis_stats = stats_summary(num_visible_per_ann, "visible_kps_per_person")
    subsection(f"Visible Keypoints per Person — {split_name}")
    for k, v in vis_stats.items():
        print(f"    {k:35s}: {v}")

    lab_stats = stats_summary(num_labeled_per_ann, "labeled_kps_per_person")
    subsection(f"Labeled Keypoints per Person — {split_name}")
    for k, v in lab_stats.items():
        print(f"    {k:35s}: {v}")

    return {
        "images": len(images),
        "annotations": len(annotations),
        "keypoint_names": kp_names,
        "fully_visible_persons": complete_persons,
        "fully_visible_percentage": pct_val(complete_persons, len(annotations)),
        "per_keypoint_visibility": kp_visibility_json,
        "visible_kps_per_person_stats": vis_stats,
        "labeled_kps_per_person_stats": lab_stats,
    }


# ============================================================================
# TRAIN vs VAL COMPARISON
# ============================================================================

def compare_splits(train_info, val_info):
    if not train_info or not val_info:
        return {}

    section("TRAIN vs VAL COMPARISON")

    metrics = [
        ("Images", "images"),
        ("Annotations", "annotations"),
        ("Categories", "categories"),
        ("Crowd Annotations", "crowd"),
    ]
    print(f"    {'Metric':30s}  {'Train':>12s}  {'Val':>12s}  {'Ratio':>8s}")
    print(f"    {'─' * 30}  {'─' * 12}  {'─' * 12}  {'─' * 8}")
    comparison_json = {}
    for label, key in metrics:
        tv = train_info.get(key, 0)
        vv = val_info.get(key, 0)
        ratio = round(tv / vv, 2) if vv else None
        print(f"    {label:30s}  {tv:>12,}  {vv:>12,}  {ratio if ratio else 'N/A':>8}")
        comparison_json[key] = {"train": tv, "val": vv, "ratio": ratio}

    # Per-category comparison
    subsection("Per-Category Split Comparison (Train vs Val)")
    all_cats = set(train_info["cat_counts"].keys()) | set(val_info["cat_counts"].keys())
    cat_id_to_name = {**train_info.get("cat_id_to_name", {}), **val_info.get("cat_id_to_name", {})}
    rows = []
    for cid in all_cats:
        tc = train_info["cat_counts"].get(cid, 0)
        vc = val_info["cat_counts"].get(cid, 0)
        rows.append((cat_id_to_name.get(cid, str(cid)), tc, vc))
    rows.sort(key=lambda r: r[1] + r[2], reverse=True)

    print(f"    {'Category':25s}  {'Train':>10s}  {'Val':>10s}  {'Total':>10s}  {'Train%':>7s}")
    print(f"    {'─' * 25}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 7}")
    per_cat_comparison = []
    for name, tc, vc in rows:
        total = tc + vc
        print(f"    {name:25s}  {tc:>10,}  {vc:>10,}  {total:>10,}  {pct(tc, total):>7s}")
        per_cat_comparison.append({
            "category": name,
            "train": tc,
            "val": vc,
            "total": total,
            "train_percentage": pct_val(tc, total),
        })

    # Size distribution comparison
    subsection("Size Distribution Comparison")
    print(f"    {'Size':10s}  {'Train':>10s}  {'Val':>10s}")
    print(f"    {'─' * 10}  {'─' * 10}  {'─' * 10}")
    size_comparison = {}
    for size in ("small", "medium", "large"):
        ts = train_info["size_dist"].get(size, 0)
        vs = val_info["size_dist"].get(size, 0)
        print(f"    {size:10s}  {ts:>10,}  {vs:>10,}")
        size_comparison[size] = {"train": ts, "val": vs}

    return {
        "overview": comparison_json,
        "per_category": per_cat_comparison,
        "size_distribution": size_comparison,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print("  COCO DATASET ANALYZER — Exhaustive Report")
    print("=" * 78)
    print(f"  Dataset root : {DATASET_ROOT}")

    if not os.path.isdir(DATASET_ROOT):
        print(f"\n  ERROR: Dataset root directory does not exist: {DATASET_ROOT}")
        print("  Please set the DATASET_ROOT variable at the top of this script,")
        print("  or pass the path as a command-line argument:")
        print("      python 1.py /path/to/COCO_ADAPTATION")
        sys.exit(1)

    files, img_dirs = discover_files(DATASET_ROOT)

    found = [k for k, v in files.items() if v]
    missing = [k for k, v in files.items() if not v]
    print(f"\n  Annotation files found   : {', '.join(found)}")
    if missing:
        print(f"  Annotation files missing : {', '.join(missing)}")

    # ── Master JSON report ──
    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "dataset_root": DATASET_ROOT,
            "annotation_files_found": found,
            "annotation_files_missing": missing,
        },
        "instances": {},
        "images": {},
        "segmentation": {},
        "captions": {},
        "keypoints": {},
        "train_vs_val": {},
        "summary": {},
    }

    train_info = None
    val_info = None

    # ── INSTANCES ──
    for split_key, split_label in [("instances_train", "Train"), ("instances_val", "Val")]:
        if files.get(split_key):
            section(f"INSTANCE ANNOTATIONS — {split_label}")
            data = load_json(files[split_key])
            info = analyze_instances(data, split_label)
            img_info = analyze_images(data, split_label, img_dirs.get(f"{split_label.lower()}2017"))
            seg_info = analyze_segmentation(data, split_label)

            # Store in report (strip non-serializable Counter objects)
            clean_info = {k: v for k, v in info.items() if k not in ("cat_counts", "cat_id_to_name", "size_dist", "crowd")}
            report["instances"][split_label.lower()] = clean_info
            report["images"][split_label.lower()] = img_info
            report["segmentation"][split_label.lower()] = seg_info

            if split_key == "instances_train":
                train_info = info
            else:
                val_info = info

    # ── CAPTIONS ──
    for split_key, split_label in [("captions_train", "Train"), ("captions_val", "Val")]:
        if files.get(split_key):
            section(f"CAPTION ANNOTATIONS — {split_label}")
            data = load_json(files[split_key])
            cap_info = analyze_captions(data, split_label)
            report["captions"][split_label.lower()] = cap_info

    # ── KEYPOINTS ──
    for split_key, split_label in [("keypoints_train", "Train"), ("keypoints_val", "Val")]:
        if files.get(split_key):
            section(f"KEYPOINT ANNOTATIONS — {split_label}")
            data = load_json(files[split_key])
            kp_info = analyze_keypoints(data, split_label)
            report["keypoints"][split_label.lower()] = kp_info

    # ── TRAIN vs VAL ──
    comparison = compare_splits(train_info, val_info)
    report["train_vs_val"] = comparison

    # ── FINAL SUMMARY ──
    section("FINAL SUMMARY")
    total_imgs = (train_info["images"] if train_info else 0) + (val_info["images"] if val_info else 0)
    total_anns = (train_info["annotations"] if train_info else 0) + (val_info["annotations"] if val_info else 0)
    total_cats = max(train_info["categories"] if train_info else 0, val_info["categories"] if val_info else 0)
    print(f"    Total images across all splits      : {fmt(total_imgs)}")
    print(f"    Total instance annotations          : {fmt(total_anns)}")
    print(f"    Total object categories             : {fmt(total_cats)}")
    print(f"    Annotation types present            : {', '.join(found)}")

    report["summary"] = {
        "total_images": total_imgs,
        "total_instance_annotations": total_anns,
        "total_categories": total_cats,
        "annotation_types": found,
    }

    # ── SAVE JSON ──
    with open(JSON_OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n    JSON report saved -> {JSON_OUTPUT_PATH}")
    print(f"    Dataset root: {DATASET_ROOT}")
    print("=" * 78)
    print("  Analysis complete.")
    print("=" * 78)


if __name__ == "__main__":
    main()
