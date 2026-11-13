"""
PyTorch implementation of the `custom_rasterizer` extension (rasterize / interpolate), for GPUs without a CUDA
toolchain (AMD ROCm on Windows, ...) and for CPU.

It reproduces `custom_rasterizer_kernel.rasterize_image` exactly: every triangle covers the pixel centres inside its
screen-space bounding box whose barycentric coordinates are all in [0, 1]; the nearest one wins through a minimum
over the same packed (quantized depth, face index + 1) keys the CUDA kernel resolves with atomicMin, so the result
does not depend on evaluation order. Barycentrics are then made perspective-correct per pixel.
"""

import torch

_MAXINT = 2147483647
_EMPTY = _MAXINT * _MAXINT + (_MAXINT - 1)  # the kernel's initial z-buffer value
_PIXELS_PER_CHUNK = 1 << 24


def _to_screen(v4: torch.Tensor, width: int, height: int):
    w = v4[:, 3:4]
    x = (v4[:, 0:1] / w * 0.5 + 0.5) * (width - 1) + 0.5
    y = (0.5 + 0.5 * v4[:, 1:2] / w) * (height - 1) + 0.5
    z = v4[:, 2:3] / w * 0.49999 + 0.5
    return torch.cat([x, y, z], dim=1)


def _barycentric(a, b, c, px, py):
    """calculateBarycentricCoordinate for points (px, py); a/b/c are (..., 2). Degenerate triangles give -1."""
    area = (c[..., 0] - a[..., 0]) * (b[..., 1] - a[..., 1]) - (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
    beta_tri = (c[..., 0] - a[..., 0]) * (py - a[..., 1]) - (px - a[..., 0]) * (c[..., 1] - a[..., 1])
    gamma_tri = (px - a[..., 0]) * (b[..., 1] - a[..., 1]) - (b[..., 0] - a[..., 0]) * (py - a[..., 1])
    degenerate = area == 0
    inv = 1.0 / torch.where(degenerate, torch.ones_like(area), area)
    beta = beta_tri * inv
    gamma = gamma_tri * inv
    alpha = 1.0 - beta - gamma
    bary = torch.stack([alpha, beta, gamma], dim=-1)
    return torch.where(degenerate[..., None], torch.full_like(bary, -1.0), bary)


@torch.no_grad()
def rasterize_image(V: torch.Tensor, F: torch.Tensor, D: torch.Tensor, width: int, height: int,
                    occlusion_truncation: float, use_depth_prior: int):
    device = V.device
    V = V.float()
    F = F.long()
    scr = _to_screen(V, width, height)  # (N, 3)
    tri = scr[F]  # (nf, 3, 3)
    x_min = tri[:, :, 0].amin(1)
    x_max = tri[:, :, 0].amax(1)
    y_min = tri[:, :, 1].amin(1)
    y_max = tri[:, :, 1].amax(1)
    # `for (int px = x_min; px < x_max + 1; ++px)`: C truncation toward zero, clipped to the image
    px0 = torch.trunc(x_min).clamp(min=0)
    py0 = torch.trunc(y_min).clamp(min=0)
    px1 = torch.minimum(torch.ceil(x_max + 1) - 1, torch.full_like(x_max, width - 1))
    py1 = torch.minimum(torch.ceil(y_max + 1) - 1, torch.full_like(y_max, height - 1))
    bw = (px1 - px0 + 1).clamp(min=0).long()
    bh = (py1 - py0 + 1).clamp(min=0).long()
    count = bw * bh

    zbuf = torch.full((height * width,), _EMPTY, dtype=torch.int64, device=device)
    depth_prior = D.float().reshape(-1) if use_depth_prior else None
    faces = torch.nonzero(count > 0, as_tuple=True)[0]
    csum = torch.cumsum(count[faces], 0)
    start = 0
    while start < faces.numel():
        base = csum[start - 1] if start > 0 else torch.zeros((), dtype=csum.dtype, device=device)
        end = int(torch.searchsorted(csum, base + _PIXELS_PER_CHUNK, right=True))
        end = max(end, start + 1)
        f = faces[start:end]
        n = count[f]
        fid = torch.repeat_interleave(f, n)
        offs = torch.arange(fid.numel(), device=device) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        px = px0[fid] + (offs % bw[fid]).float()
        py = py0[fid] + torch.div(offs, bw[fid], rounding_mode="floor").float()
        t = tri[fid]
        bary = _barycentric(t[:, 0, :2], t[:, 1, :2], t[:, 2, :2], px + 0.5, py + 0.5)
        inside = ((bary >= 0) & (bary <= 1)).all(-1)
        depth = (bary * t[:, :, 2]).sum(-1)
        pixel = py.long() * width + px.long()
        if depth_prior is not None:
            inside &= depth >= depth_prior[pixel] * 0.49999 + 0.5 + occlusion_truncation
        z = torch.trunc(depth * (2 << 17)).long()
        inside &= z >= 0  # a negative quantized depth wraps to a huge unsigned key that never wins in the kernel
        token = z * _MAXINT + (fid + 1)
        zbuf.scatter_reduce_(0, pixel[inside], token[inside], reduce="amin")
        start = end

    f_all = zbuf % _MAXINT
    empty = f_all == _MAXINT - 1
    findices = torch.where(empty, torch.zeros_like(f_all), f_all).to(torch.int32).view(height, width)

    # perspective-correct barycentrics of the winning face at each covered pixel
    barycentric = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    covered = torch.nonzero(findices.view(-1) > 0, as_tuple=True)[0]
    if covered.numel():
        f = findices.view(-1)[covered].long() - 1
        v = V[F[f]]  # (n, 3, 4)
        w = v[:, :, 3]
        sx = (v[:, :, 0] / w * 0.5 + 0.5) * (width - 1) + 0.5
        sy = (0.5 + 0.5 * v[:, :, 1] / w) * (height - 1) + 0.5
        s = torch.stack([sx, sy], dim=-1)
        px = (covered % width).float() + 0.5
        py = torch.div(covered, width, rounding_mode="floor").float() + 0.5
        b = _barycentric(s[:, 0], s[:, 1], s[:, 2], px, py) / w
        b = b / b.sum(-1, keepdim=True)
        barycentric[covered] = b
    return findices, barycentric.view(height, width, 3)


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    assert pos.device == tri.device
    return rasterize_image(pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior)


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
