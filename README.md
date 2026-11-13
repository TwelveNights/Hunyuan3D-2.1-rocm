# Hunyuan3D-2.1 for AMD GPUs (ROCm)

Image-to-3D with PBR textures from [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) on AMD Radeon
and Instinct GPUs, on Linux and on Windows. Upstream's texture painter needs a CUDA rasterizer, a C++ extension and
Blender; this fork replaces all three with PyTorch, NumPy and trimesh, so both stages, shape and texture, run on ROCm
PyTorch with nothing to compile, and texturing fits a 16 GB card.

![Hunyuan3D-2.1 on an RX 7800 XT: input image and four views of the textured model](assets/rocm/rx7800xt-hunyuan3d-2.1.jpg)

*Input image and the textured model Hunyuan3D-2.1 made from it, on a Radeon RX 7800 XT 16 GB under Windows: shape
40 s, PBR texture 96 s (12.9 GiB peak).*

- **Faithful replacements, checked against the originals.** The PyTorch rasterizer returns the same face ids as a
  HIP build of the CUDA kernel (100% on real meshes); the NumPy inpainting matches the C++ module.
- **Fits 16 GB.** On ROCm 7.x PyTorch the paint model's attention fell back to a kernel that materializes the whole
  score matrix (21 GiB peak); an exact rewrite brings it to 10 GiB.
- **Much faster.** On a Radeon PRO W7900: shape 135 s -> 26.5 s (FlashVDM decoder,
  [SageAttention-rocm](https://github.com/TwelveNights/SageAttention-rocm), 30 steps), texture 150 s -> 46 s.

## Quick start

In ComfyUI, use the [ComfyUI-3D-rocm](https://github.com/TwelveNights/ComfyUI-3D-rocm) node pack instead: the Hunyuan3D-2.1 painter as nodes, installed from ComfyUI Manager.

Install a ROCm build of PyTorch first (Linux: `pip install torch torchvision --index-url
https://download.pytorch.org/whl/rocm7.0`; Windows: AMD's ROCm wheels), then:

```bash
git clone -b rocm https://github.com/TwelveNights/Hunyuan3D-2.1-rocm.git && cd Hunyuan3D-2.1-rocm
pip install diffusers transformers accelerate einops huggingface_hub opencv-python-headless trimesh omegaconf scikit-image pymeshlab xatlas pygltflib rembg onnxruntime timm torchdiffeq pytorch-lightning spandrel
python demo_rocm.py assets/demo.png output     # -> output/demo_textured.glb
```

`demo_rocm.py` uses the fast settings (FlashVDM, 30 steps, SageAttention when installed); `--no-texture` for shape
only. The first run of each stage is slow while MIOpen tunes convolutions; later runs are fast.

**[ROCM.md](ROCM.md)**: what changed and how it was verified, measurements, Windows notes.
Upstream's documentation: [README_upstream.md](README_upstream.md).

## Related

- [ComfyUI-3D-rocm](https://github.com/TwelveNights/ComfyUI-3D-rocm): TripoSG and Hunyuan3D-2.1 Paint as ComfyUI nodes on AMD
- [TripoSG-rocm](https://github.com/TwelveNights/TripoSG-rocm): TripoSG image-to-3D on AMD
- [nunchaku-rocm](https://github.com/TwelveNights/nunchaku-rocm): 4-bit FLUX, Qwen-Image, Z-Image and SDXL on AMD
- [SageAttention-rocm](https://github.com/TwelveNights/SageAttention-rocm): 8-bit attention for AMD GPUs

License: the Tencent Hunyuan 3D community license applies (it excludes the EU, the UK and South Korea).
