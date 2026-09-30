# Save as app_8005.py inside the installed TRELLIS.2 repository.
#
# Run on H100:
#   conda activate trellis2
#   hf auth login
#   CUDA_VISIBLE_DEVICES=0 python app_8005.py
#
# Hugging Face access required:
# https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#
# On the Mac:
# ssh -i "/Users/asadirfan358/Downloads/untitled folder/ubuntu-22-carla_key.pem" \
#   -N -L 8005:127.0.0.1:8005 azureuser@4.182.250.227
#
# Open: http://localhost:8005
#
# IMPORTANT:
# TRELLIS.2 is officially a single-image model.
# Multiple images below use experimental prediction averaging.
# This does not perform camera-calibrated multi-view reconstruction.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn"

import gc
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline, samplers
from trellis2.pipelines.base import Pipeline
from trellis2.modules import image_feature_extractor

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

OUTPUT_DIR = ROOT / "outputs_8005"
OUTPUT_DIR.mkdir(exist_ok=True)

LOCK = threading.Lock()
pipeline = None


def load_model():
    """Load native checkpoint precision, without quantization or CPU offload."""
    global pipeline

    if pipeline is not None:
        return pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    # Use the upstream model loader, omitting the unused background remover.
    loaded = Pipeline.from_pretrained.__func__(
        Trellis2ImageTo3DPipeline,
        "microsoft/TRELLIS.2-4B",
    )
    config = loaded._pretrained_args

    options = {}
    for name in (
        "sparse_structure_sampler",
        "shape_slat_sampler",
        "tex_slat_sampler",
    ):
        settings = config[name]
        options[name] = getattr(samplers, settings["name"])(
            **settings["args"]
        )
        options[name + "_params"] = settings["params"]

    encoder = config["image_cond_model"]

    model = Trellis2ImageTo3DPipeline(
        models=loaded.models,
        **options,
        shape_slat_normalization=config["shape_slat_normalization"],
        tex_slat_normalization=config["tex_slat_normalization"],
        image_cond_model=getattr(
            image_feature_extractor, encoder["name"]
        )(**encoder["args"]),
        rembg_model=None,
        low_vram=False,
        default_pipeline_type="1024_cascade",
    )

    model.cuda()
    pipeline = model

    print("Loaded on:", torch.cuda.get_device_name(0))
    return pipeline


def prepare_image(path):
    """Keep existing segmentation and composite transparency onto black."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")

    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    image = Image.alpha_composite(background, image).convert("RGB")

    # Crop using non-black bounds; retain dark pixels inside the object.
    pixels = np.asarray(image)
    foreground = pixels.max(axis=2) > 8

    if not foreground.any():
        raise ValueError(f"No visible object in {Path(path).name}")

    y, x = np.nonzero(foreground)
    image = image.crop(
        (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
    )

    side = max(16, int(max(image.size) * 1.2))
    square = Image.new("RGB", (side, side), "black")
    square.paste(
        image,
        ((side - image.width) // 2, (side - image.height) // 2),
    )
    return square


class ViewConditions:
    def __init__(self, conditions):
        self.conditions = conditions


@contextmanager
def use_multiple_images(model, images):
    """Average per-view guided predictions for one shared 3D latent."""
    if len(images) == 1:
        yield
        return

    saved = []
    cache = {}
    original_get_cond = model.get_cond

    def replace(obj, name, function):
        saved.append(
            (obj, name, name in obj.__dict__, obj.__dict__.get(name))
        )
        setattr(obj, name, MethodType(function, obj))

    def get_cond(self, unused_image, resolution, include_neg_cond=True):
        if resolution not in cache:
            conditions = [
                original_get_cond([image], resolution)
                for image in images
            ]
            cache[resolution] = {
                "cond": ViewConditions(conditions),
                "neg_cond": conditions[0]["neg_cond"],
            }
        return cache[resolution]

    def make_predictor(original):
        def predict(self, flow_model, x_t, t, cond=None, **kwargs):
            if not isinstance(cond, ViewConditions):
                return original(flow_model, x_t, t, cond, **kwargs)

            result = None
            count = len(cond.conditions)

            for view in cond.conditions:
                options = dict(kwargs)
                options["neg_cond"] = view["neg_cond"]

                prediction = original(
                    flow_model, x_t, t, view["cond"], **options
                ) / count

                result = (
                    prediction if result is None else result + prediction
                )

            return result

        return predict

    try:
        replace(model, "get_cond", get_cond)

        for name in (
            "sparse_structure_sampler",
            "shape_slat_sampler",
            "tex_slat_sampler",
        ):
            sampler = getattr(model, name)
            replace(
                sampler,
                "_inference_model",
                make_predictor(sampler._inference_model),
            )

        yield
    finally:
        for obj, name, existed, previous in reversed(saved):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)


def generate(files, resolution, seed, progress=gr.Progress()):
    if not files:
        raise gr.Error("Upload images of the same object.")

    if len(files) > 8:
        raise gr.Error("Upload at most 8 images.")

    folder = OUTPUT_DIR / uuid.uuid4().hex
    folder.mkdir()

    outputs = mesh = glb = None

    try:
        images = [prepare_image(path) for path in files]

        for index, image in enumerate(images):
            image.save(folder / f"input_{index + 1}.png")

        with LOCK:
            progress(0.05, desc="Loading TRELLIS.2")
            model = load_model()

            progress(
                0.15,
                desc=f"Generating one object using {len(images)} image(s)",
            )

            pipeline_type = {
                "512": "512",
                "1024": "1024_cascade",
                "1536": "1536_cascade",
            }[str(resolution)]

            with torch.no_grad(), use_multiple_images(model, images):
                outputs = model.run(
                    images[0],
                    seed=int(seed),
                    num_samples=1,
                    preprocess_image=False,
                    pipeline_type=pipeline_type,
                )

            mesh = outputs[0]
            mesh.simplify(16777216)

            progress(0.8, desc="Exporting mesh and 4K textures")

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000,
                texture_size=4096,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )

            output_path = folder / "object.glb"
            glb.export(str(output_path))

        (folder / "settings.json").write_text(
            json.dumps(
                {
                    "model": "microsoft/TRELLIS.2-4B",
                    "views": len(images),
                    "seed": int(seed),
                    "resolution": str(resolution),
                    "quantization": None,
                    "multi_image_method": "experimental prediction averaging",
                },
                indent=2,
            )
        )

        progress(1, desc="Finished")
        return (
            str(output_path),
            str(output_path),
            f"Finished. Saved to {output_path}",
        )

    except Exception as error:
        logging.exception("Generation failed")
        raise gr.Error(str(error)) from error

    finally:
        outputs = mesh = glb = None
        gc.collect()
        torch.cuda.empty_cache()


with gr.Blocks(title="TRELLIS.2 — H100") as app:
    gr.Markdown(
        "# TRELLIS.2 — Image to 3D\n"
        "Upload segmented images of the same object on black backgrounds.\n\n"
        "**No quantization.** Multiple views use experimental prediction "
        "averaging; exact geometric alignment is not guaranteed."
    )

    files = gr.File(
        label="Object images",
        file_count="multiple",
        file_types=["image"],
        type="filepath",
    )

    with gr.Row():
        resolution = gr.Dropdown(
            choices=["512", "1024", "1536"],
            value="1024",
            label="Resolution",
        )
        seed = gr.Number(value=42, precision=0, label="Seed")

    button = gr.Button("Generate 3D object", variant="primary")
    viewer = gr.Model3D(label="3D model", height=600)
    download = gr.File(label="Download GLB")
    status = gr.Textbox(label="Status", interactive=False)

    button.click(
        generate,
        inputs=[files, resolution, seed],
        outputs=[viewer, download, status],
        concurrency_limit=1,
    )


if __name__ == "__main__":
    app.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=8005,
        share=False,
        allowed_paths=[str(OUTPUT_DIR)],
    )
