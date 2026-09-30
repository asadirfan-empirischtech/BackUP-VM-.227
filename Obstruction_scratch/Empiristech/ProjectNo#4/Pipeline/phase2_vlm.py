"""
phase2_vlm.py — Spatial Reasoning with Qwen-VL

Phase 2 of the Hazard Zone Segmentation Pipeline.

Purpose:
    Sends BOTH the original (clean context) and gridded (spatial reference)
    images to a locally-served Qwen-VL model via the OpenAI-compatible
    chat/completions endpoint.

    The VLM reads the grid IDs directly from the overlay image and classifies
    each relevant cell as floor (inside zone), boundary (on the perimeter),
    or exclusion (outside / machinery).

    This dual-image prompting strategy is the core innovation: the original
    image provides unobstructed scene semantics, while the gridded image
    provides unambiguous spatial anchors that the VLM can reference by name
    in its structured JSON output.

    NOTE: Uses raw `requests` library for the OpenAI-compatible API call
    to avoid the langchain/openai dependency. This works with any vLLM,
    Ollama, or compatible server.

Output:
    vlm_response.json — JSON with floor_points, boundary_points, exclusion_points

Usage (standalone):
    python phase2_vlm.py --original /path/to/original.png \\
                         --gridded /path/to/gridded.jpg \\
                         --grid-map /path/to/grid_map.json
"""

import argparse
import base64
import json
import os
import re
import sys

import requests

# ── Import pipeline configuration ───────────────────────────
try:
    from config import (
        OUTPUT_DIR,
        QWEN_API_URL,
        QWEN_MODEL,
        QWEN_API_KEY,
        QWEN_MAX_TOKENS,
        QWEN_TEMPERATURE,
        QWEN_MAX_RETRIES,
        VLM_RESPONSE_FILENAME,
    )
except ImportError:
    from Pipeline.config import (
        OUTPUT_DIR,
        QWEN_API_URL,
        QWEN_MODEL,
        QWEN_API_KEY,
        QWEN_MAX_TOKENS,
        QWEN_TEMPERATURE,
        QWEN_MAX_RETRIES,
        VLM_RESPONSE_FILENAME,
    )


# ══════════════════════════════════════════════════════════════
#  Helper: Encode an image file to base64
# ══════════════════════════════════════════════════════════════
def _encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ══════════════════════════════════════════════════════════════
#  Helper: Robust JSON extraction from VLM output
# ══════════════════════════════════════════════════════════════
def _extract_json(response_text: str, attempt: int = 0) -> dict:
    """
    Extract a JSON object from potentially noisy VLM output.

    Handles:
      - </think> reasoning blocks (Qwen thinking mode)
      - Markdown ```json ... ``` code fences
      - Stray '...' placeholder text
      - Greedy curly-brace extraction as a final fallback
    """
    raw = response_text

    # Strip thinking blocks
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    # Remove literal '...' that Qwen sometimes echoes from the prompt
    raw = raw.replace("...", "")

    # Attempt 1: Markdown code fence
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # Attempt 2: Greedy outermost { ... }
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Attempt 3: Raw parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Save the raw output for debugging
        debug_path = os.path.join(OUTPUT_DIR, f"vlm_raw_output_attempt_{attempt}.txt")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(debug_path, "w") as dbg:
            dbg.write(response_text)
        raise ValueError(
            f"Could not extract valid JSON from VLM response (attempt {attempt + 1}). "
            f"Raw output saved to {debug_path}"
        ) from exc


# ══════════════════════════════════════════════════════════════
#  Helper: Detect MIME type from file extension
# ══════════════════════════════════════════════════════════════
def _mime_type(path: str) -> str:
    """Return the MIME type string for common image extensions."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(ext, "image/png")


# ══════════════════════════════════════════════════════════════
#  Helper: Build the structured VLM prompt
# ══════════════════════════════════════════════════════════════
def _build_prompt(grid_ids: list[str]) -> str:
    """
    Construct the instruction prompt for Qwen-VL.

    Provides:
      - Clear task description and context
      - Exact output schema
      - The set of valid grid IDs to constrain VLM vocabulary
    """

    # Condense the valid ID list into a range description to save tokens
    # e.g. "A1–A40, B1–B40, ... AD1–AD40"
    rows_seen = {}
    for gid in grid_ids:
        # Split at the boundary between letters and digits
        split_idx = 0
        for i, ch in enumerate(gid):
            if ch.isdigit():
                split_idx = i
                break
        row_part = gid[:split_idx]
        col_part = gid[split_idx:]
        rows_seen.setdefault(row_part, []).append(int(col_part))

    # Build compact range strings
    range_parts = []
    for row_label in sorted(rows_seen.keys(), key=lambda r: (len(r), r)):
        cols = sorted(rows_seen[row_label])
        range_parts.append(f"{row_label}{cols[0]}–{row_label}{cols[-1]}")

    id_range_str = ", ".join(range_parts)

    prompt = f"""You are an expert industrial computer-vision spatial analyst.

You are given TWO images of the same manufacturing floor scene:
  • IMAGE 1 (original): The raw, unmodified photograph — use this for clear visual context.
  • IMAGE 2 (gridded):  The same scene overlaid with a labeled alphanumeric grid.
    Each grid cell has a unique ID printed at its center (e.g. A1, B12, C5).

YOUR TASK:
1. Identify every restricted hazard zone bounded by floor paintings (yellow hatched lines, warning stripes), warning tape, or physical barriers (safety cones, barricades).
2. For EACH grid cell ID visible in Image 2, classify it based on its physical location in the scene.

RETURN a strictly formatted JSON object with exactly three arrays:

{{
  "floor_points": ["<id>", ...],
  "boundary_points": ["<id>", ...],
  "exclusion_points": ["<id>", ...]
}}

Definitions:
  • floor_points:     Grid IDs whose centers fall on EMPTY FLOOR strictly INSIDE the restricted hazard boundary. This is the open floor area enclosed by the safety markings.
  • boundary_points:  Grid IDs whose centers fall directly ON the perimeter lines, painted markings, warning tape, safety cones, or barricade edges.
  • exclusion_points: Grid IDs whose centers fall on equipment, machinery, safe walkway pathways, walls, or ANY area OUTSIDE the restricted zone.

RULES:
  - Use ONLY IDs from this valid set: {id_range_str}
  - Be thorough — classify as many grid cells as you can see, especially those inside and on the boundary.
  - If you see MULTIPLE separate hazard zones, merge all their floor/boundary/exclusion points into the same three arrays.

OUTPUT FORMAT:
  - First, do your spatial reasoning BRIEFLY inside <think>...</think> tags.
  - Then, IMMEDIATELY after the closing </think> tag, output ONLY the raw JSON object.
  - Do NOT include any text, explanation, or markdown outside the JSON structure after </think>.
  - Keep your thinking concise — focus on identifying zone boundaries, not calculating pixel math.
"""
    return prompt


# ══════════════════════════════════════════════════════════════
#  Helper: Call the OpenAI-compatible chat/completions endpoint
# ══════════════════════════════════════════════════════════════
def _call_qwen_api(prompt_text: str, b64_original: str, mime_original: str,
                   b64_gridded: str, mime_gridded: str) -> str:
    """
    Make a raw HTTP POST to the vLLM OpenAI-compatible /chat/completions
    endpoint with multi-image content.

    Returns the assistant's response text.
    """
    url = f"{QWEN_API_URL}/chat/completions"

    payload = {
        "model": QWEN_MODEL,
        "max_tokens": QWEN_MAX_TOKENS,
        "temperature": QWEN_TEMPERATURE,
        # Enable Qwen3's native thinking mode so reasoning goes into
        # <think> tags and the final answer is clean JSON
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_original};base64,{b64_original}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_gridded};base64,{b64_gridded}"},
                    },
                ],
            }
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QWEN_API_KEY}",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()

    data = resp.json()

    # Extract the assistant message content
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"Empty 'choices' in API response: {json.dumps(data)[:500]}")

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Empty content in API response: {json.dumps(data)[:500]}")

    return content


# ══════════════════════════════════════════════════════════════
#  Core Function: Analyze Scene with Qwen-VL
# ══════════════════════════════════════════════════════════════
def analyze_with_vlm(
    original_img_path: str,
    gridded_img_path: str,
    grid_map: dict,
) -> dict:
    """
    Send dual images to Qwen-VL and receive classified grid IDs.

    Parameters
    ----------
    original_img_path : str
        Path to the original (unmodified) manufacturing floor image.
    gridded_img_path : str
        Path to the grid-overlaid image produced by Phase 1.
    grid_map : dict
        The {id: [x, y]} dictionary from Phase 1 (used to build the valid ID set).

    Returns
    -------
    dict
        Parsed JSON with keys: floor_points, boundary_points, exclusion_points
    """

    # ── Validate inputs ─────────────────────────────────────
    for label, path in [("Original", original_img_path), ("Gridded", gridded_img_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} image not found: {path}")

    grid_ids = sorted(grid_map.keys(), key=lambda g: (len(g), g))
    print(f"[Phase 2] Valid grid IDs: {len(grid_ids)} (from {grid_ids[0]} to {grid_ids[-1]})")

    # ── Encode images ───────────────────────────────────────
    print("[Phase 2] Encoding images to base64...")
    b64_original = _encode_image_base64(original_img_path)
    b64_gridded = _encode_image_base64(gridded_img_path)
    mime_original = _mime_type(original_img_path)
    mime_gridded = _mime_type(gridded_img_path)

    # ── Build the prompt ────────────────────────────────────
    prompt_text = _build_prompt(grid_ids)

    # ── Call Qwen-VL with retries ───────────────────────────
    vlm_result = None
    for attempt in range(QWEN_MAX_RETRIES):
        print(f"\n[Phase 2] ── VLM Attempt {attempt + 1}/{QWEN_MAX_RETRIES} ──")
        try:
            response_text = _call_qwen_api(
                prompt_text, b64_original, mime_original, b64_gridded, mime_gridded
            )
            print(f"[Phase 2] Response length: {len(response_text)} chars")

            parsed = _extract_json(response_text, attempt=attempt)

            # Validate required keys
            required_keys = {"floor_points", "boundary_points", "exclusion_points"}
            missing = required_keys - set(parsed.keys())
            if missing:
                print(f"[Phase 2] ⚠️  Missing keys: {missing}. Retrying...")
                continue

            # Basic type validation
            valid_structure = True
            for key in required_keys:
                if not isinstance(parsed[key], list):
                    print(f"[Phase 2] ⚠️  '{key}' is not a list. Retrying...")
                    valid_structure = False
                    break
            if not valid_structure:
                continue

            # Log summary
            fp = len(parsed["floor_points"])
            bp = len(parsed["boundary_points"])
            ep = len(parsed["exclusion_points"])
            print(f"[Phase 2] ✅ Classified: {fp} floor | {bp} boundary | {ep} exclusion")

            vlm_result = parsed
            break

        except requests.exceptions.ConnectionError:
            print(f"[Phase 2] ❌ Could not connect to Qwen-VL at {QWEN_API_URL}")
        except requests.exceptions.Timeout:
            print("[Phase 2] ❌ Qwen-VL request timed out (300s)")
        except Exception as e:
            print(f"[Phase 2] ❌ Attempt {attempt + 1} failed: {e}")

    if vlm_result is None:
        raise RuntimeError(
            f"Qwen-VL failed to return valid JSON after {QWEN_MAX_RETRIES} attempts. "
            f"Check debug files in {OUTPUT_DIR}/"
        )

    # ── Save response ───────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    response_path = os.path.join(OUTPUT_DIR, VLM_RESPONSE_FILENAME)
    with open(response_path, "w") as f:
        json.dump(vlm_result, f, indent=2)
    print(f"[Phase 2] Saved VLM response → {response_path}")

    return vlm_result


# ══════════════════════════════════════════════════════════════
#  Standalone Entry Point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 — Spatial reasoning with Qwen-VL (dual-image prompting)"
    )
    parser.add_argument("--original", "-o", required=True, help="Path to original image")
    parser.add_argument("--gridded", "-g", required=True, help="Path to gridded image")
    parser.add_argument("--grid-map", "-m", required=True, help="Path to grid_map.json")

    args = parser.parse_args()

    try:
        with open(args.grid_map, "r") as f:
            gmap = json.load(f)
        result = analyze_with_vlm(args.original, args.gridded, gmap)
        print(f"\n✅ Phase 2 complete. Classified {sum(len(v) for v in result.values())} grid cells.")
    except Exception as e:
        print(f"\n❌ Phase 2 failed: {e}")
        sys.exit(1)
