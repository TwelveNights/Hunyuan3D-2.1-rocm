# Hunyuan3D-2.1 on AMD GPUs (ROCm, Linux and Windows)

This branch runs both stages, shape generation and PBR texture painting, without CUDA or C++ extensions and without
Blender, so it works on AMD GPUs with ROCm PyTorch, including Windows where nothing can be compiled. Texture
painting fits 16 GB cards.

## What changed

| Upstream component | Here | Check against the original |
| --- | --- | --- |
| `custom_rasterizer` (CUDA kernel) | `hy3dpaint/DifferentiableRenderer/rasterizer_torch.py`, used when the extension is not built | Face ids 100% identical to a HIP build of the original on real meshes (4 views, with and without depth prior); barycentrics within 5e-4 |
| `mesh_inpaint_processor` (C++) | `hy3dpaint/DifferentiableRenderer/mesh_inpaint_numpy.py` | `forward` identical; `smooth` fills the same texels, colors within 0.008 mean (vectorized passes instead of in-place updates) |
| `bpy` (Blender) for OBJ -> GLB | trimesh, with a glTF PBR material (base color, metallic-roughness, normal) | |
| transformers 5 | Dinov2 attention weights renamed on load | |
| current diffusers / trimesh | `trust_remote_code` for the shipped paint pipeline; `simplify_quadric_decimation(face_count=)` | |

## Memory and speed

The paint UNet's multiview reference attention packs both materials into the values (value head dim 128, query
64). Fused attention kernels need equal head dims, so PyTorch fell back to the math kernel and materialized a
5.6 GiB score matrix per call: a 21.3 GiB peak at 512 px with 6 views. Attending each value slice separately is
exact (the weights do not depend on the values) and brings the UNet peak to 10.0 GiB. The UNet code is loaded from
the model snapshot, so the fix is applied after loading (`hy3dpaint/utils/attention_utils.py`). The 12 view VAE
encodes/decodes are sliced.

Radeon PRO W7900 (gfx1100), chair image, warm:

| Stage | Upstream settings | Optimized | How |
| --- | ---: | ---: | --- |
| Shape | 135 s | **26.5 s** | `pipeline.enable_flashvdm(mc_algo="mc")`, SageAttention, `num_inference_steps=30` (mean surface distance to the default: 0.35% of the bounding-box diagonal) |
| Paint (6 views, 512) | 150 s, 21.3 GiB | **46 s**, 10 GiB | the attention fix + VAE slicing (64 s), + SageAttention |

First runs are much slower while MIOpen tunes convolutions; set `MIOPEN_FIND_MODE=FAST` to skip the exhaustive
search. SageAttention: [SageAttention-rocm](https://github.com/TwelveNights/SageAttention-rocm), enable it with
`from rocm_attention import use_sage_attention; use_sage_attention()` (repo root) before loading the pipelines.

## Install

Install a ROCm build of PyTorch first, then (without replacing it):

```
pip install diffusers transformers accelerate einops huggingface_hub opencv-python-headless trimesh omegaconf scikit-image pymeshlab xatlas pygltflib rembg onnxruntime timm torchdiffeq pytorch-lightning realesrgan basicsr
```

Download `RealESRGAN_x4plus.pth` to `hy3dpaint/ckpt/` and run `demo.py`. If the Hugging Face cache is not writable,
set `HF_MODULES_CACHE` (diffusers copies the paint pipeline's code there).

License: the Tencent Hunyuan 3D community license applies to the models and code (it excludes the EU, the UK and
South Korea).
