"""
Image -> textured (PBR) GLB with Hunyuan3D-2.1 on AMD GPUs, with the fast settings from ROCM.md.

  python demo_rocm.py image.png [out_dir] [--steps 30] [--views 6] [--resolution 512] [--no-texture] [--no-sage]

Works without CUDA/C++ extensions or Blender (see ROCM.md); runs on NVIDIA and CPU-less setups the same way.
"""

import argparse
import os
import sys

os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")  # skip MIOpen's exhaustive first-run convolution tuning
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "hy3dshape"), os.path.join(ROOT, "hy3dpaint")]  # embedded Pythons ignore the script dir

try:
    from torchvision_fix import apply_fix

    apply_fix()
except Exception as e:  # noqa: BLE001
    print(f"torchvision fix not applied: {e}")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("out_dir", nargs="?", default="output")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--views", type=int, default=6)
ap.add_argument("--resolution", type=int, default=512)
ap.add_argument("--no-texture", action="store_true")
ap.add_argument("--no-sage", action="store_true")
args = ap.parse_args()

if not args.no_sage:
    from rocm_attention import use_sage_attention

    print("SageAttention:", "on" if use_sage_attention() else "not installed, using PyTorch attention")

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
from hy3dshape.rembg import BackgroundRemover  # noqa: E402

os.makedirs(args.out_dir, exist_ok=True)
name = os.path.splitext(os.path.basename(args.image))[0]
image = Image.open(args.image)
if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
    image = BackgroundRemover()(image.convert("RGB"))
image_path = os.path.join(args.out_dir, f"{name}_rgba.png")
image.save(image_path)

shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
shape.enable_flashvdm(mc_algo="mc")
mesh = shape(image=image, num_inference_steps=args.steps, generator=torch.manual_seed(0))[0]
shape_path = os.path.join(args.out_dir, f"{name}_shape.glb")
mesh.export(shape_path)
print("shape:", shape_path)
del shape
torch.cuda.empty_cache()

if not args.no_texture:
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    conf = Hunyuan3DPaintConfig(args.views, args.resolution)
    conf.realesrgan_ckpt_path = os.path.join(ROOT, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(ROOT, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(ROOT, "hy3dpaint", "hunyuanpaintpbr")
    if not os.path.exists(conf.realesrgan_ckpt_path):
        os.makedirs(os.path.dirname(conf.realesrgan_ckpt_path), exist_ok=True)
        torch.hub.download_url_to_file(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            conf.realesrgan_ckpt_path,
        )
    out = Hunyuan3DPaintPipeline(conf)(
        mesh_path=shape_path, image_path=image_path, output_mesh_path=os.path.join(args.out_dir, f"{name}_textured.obj")
    )
    print("textured:", out.replace(".obj", ".glb"))
