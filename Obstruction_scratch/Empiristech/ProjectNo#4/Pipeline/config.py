"""
config.py — Centralized Configuration for the Hazard Zone Segmentation Pipeline.

All tunable parameters live here so that every phase module imports from a
single source of truth.  Override any value via environment variables or by
editing the constants below.

Project Context
---------------
This pipeline bridges semantic scene understanding (Qwen-VL) with pixel-perfect
geometric segmentation (SAM 3) to extract the exact floor area of restricted
hazard zones on manufacturing floors, while strictly excluding safe pathways
and intersecting machinery.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Directory Paths
# ──────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent
OUTPUT_DIR = PIPELINE_DIR / "outputs"
TEST_IMAGES_DIR = PROJECT_DIR / "Testing Images"

# ──────────────────────────────────────────────────────────────
# Qwen-VL Model Configuration (served via vLLM OpenAI-compat)
# ──────────────────────────────────────────────────────────────
QWEN_API_URL = os.getenv("QWEN_API_URL", "http://localhost:8000/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3.8-27B")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "secret-123")
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "16384"))
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.15"))
QWEN_MAX_RETRIES = int(os.getenv("QWEN_MAX_RETRIES", "3"))

# ──────────────────────────────────────────────────────────────
# SAM 3 Segmentation API
# ──────────────────────────────────────────────────────────────
SAM3_API_URL = os.getenv("SAM3_API_URL", "http://localhost:8001/v1/segment")
SAM3_MAX_RETRIES = int(os.getenv("SAM3_MAX_RETRIES", "4"))
SAM3_RETRY_DELAY = float(os.getenv("SAM3_RETRY_DELAY", "2.0"))

# ──────────────────────────────────────────────────────────────
# Phase 1 — Grid Overlay Settings
# ──────────────────────────────────────────────────────────────
# Grid density (configurable — medium density is the default)
GRID_COLS = int(os.getenv("GRID_COLS", "40"))    # Number of vertical divisions
GRID_ROWS = int(os.getenv("GRID_ROWS", "30"))    # Number of horizontal divisions

# Visual tuning
GRID_LINE_COLOR = (180, 180, 180)   # Light gray (BGR)
GRID_LINE_THICKNESS = 1
GRID_OPACITY = float(os.getenv("GRID_OPACITY", "0.35"))

# Label rendering — white text with black outline for universal readability
GRID_LABEL_COLOR = (255, 255, 255)         # White fill (BGR)
GRID_LABEL_OUTLINE_COLOR = (0, 0, 0)       # Black outline (BGR)
GRID_LABEL_OUTLINE_THICKNESS = 2
GRID_LABEL_FONT_SCALE = float(os.getenv("GRID_FONT_SCALE", "0.32"))

# ──────────────────────────────────────────────────────────────
# Phase 4 — Overlay Visualization
# ──────────────────────────────────────────────────────────────
# Color for the segmentation mask overlay (BGR format)
OVERLAY_COLOR_BGR = (0, 0, 255)   # Red highlight
OVERLAY_ALPHA = float(os.getenv("OVERLAY_ALPHA", "0.45"))

# ──────────────────────────────────────────────────────────────
# Output Filenames (saved inside OUTPUT_DIR)
# ──────────────────────────────────────────────────────────────
GRIDDED_IMAGE_FILENAME = "gridded_image.jpg"
GRID_MAP_FILENAME = "grid_map.json"
VLM_RESPONSE_FILENAME = "vlm_response.json"
SAM_PROMPTS_FILENAME = "sam_prompts.json"
SEGMENTATION_MASK_FILENAME = "segmentation_mask.png"
FINAL_OVERLAY_FILENAME = "final_overlay.jpg"
