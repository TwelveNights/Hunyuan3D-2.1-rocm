"""
NumPy implementation of the `mesh_inpaint_processor` extension (meshVerticeInpaint), so texture baking needs no C++
build. It follows the C++ step for step: vertex colors come from the texels their UVs land on, uncolored vertices take
the inverse-square-distance weighted mean of their colored mesh neighbours until the uncolored count stops changing
("smooth"), or colors spread outward ring by ring ("forward"); colored vertices are written back to their texels.

One difference: the C++ "smooth" loop updates vertices one after another within a pass (later ones see earlier
updates), while this updates all of them from the previous pass's colors at once. The set of texels filled is the
same; their colors can differ slightly. They only fill texels no view saw.
"""

import numpy as np


def _round_half_away(x):
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5)).astype(np.int64)


def _setup(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx):
    h, w, c = texture.shape
    corner_v = pos_idx.reshape(-1).astype(np.int64)
    corner_uv = uv_idx.reshape(-1).astype(np.int64)
    u = _round_half_away((1.0 - vtx_uv[corner_uv, 1].astype(np.float64)) * (h - 1))
    v = _round_half_away(vtx_uv[corner_uv, 0].astype(np.float64) * (w - 1))
    texel = u * w + v
    seen = mask.reshape(-1)[texel] > 0
    nv = vtx_pos.shape[0]
    color = np.zeros((nv, c), np.float32)
    colored = np.zeros(nv, bool)
    # in corner order, so a vertex shared by several faces keeps the last visible corner's texel, as in C++
    color[corner_v[seen]] = texture.reshape(-1, c)[texel[seen]]
    colored[corner_v[seen]] = True
    # G[a] gets b for every face edge a -> b in winding order (duplicates kept, as in C++)
    f = pos_idx.astype(np.int64)
    src = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    dst = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    d = np.linalg.norm(vtx_pos[src].astype(np.float64) - vtx_pos[dst].astype(np.float64), axis=1)
    weight = (1.0 / np.maximum(d, 1e-4)) ** 2
    return h, w, c, corner_v, texel, seen, color, colored, src, dst, weight


def _write_back(texture, mask, color, colored, corner_v, texel, c):
    new_texture = texture.reshape(-1, c).copy()
    new_mask = mask.reshape(-1).copy()
    sel = colored[corner_v]
    new_texture[texel[sel]] = color[corner_v[sel]]
    new_mask[texel[sel]] = 255
    return new_texture, new_mask


def _smooth(color, colored, corner_v, seen, src, dst, weight, nv):
    uncolored_list = corner_v[~seen]  # one entry per uncolored face corner, as the C++ list
    targets = np.unique(uncolored_list)
    edge_in = np.isin(src, targets)
    src, dst, weight = src[edge_in], dst[edge_in], weight[edge_in]
    smooth_count, last = 2, 0
    while smooth_count > 0:
        wgt = weight * colored[dst]
        total = np.bincount(src, weights=wgt, minlength=nv)
        sums = np.stack([np.bincount(src, weights=wgt * color[dst, k], minlength=nv) for k in range(color.shape[1])], 1)
        ok = total[targets] > 0
        t = targets[ok]
        color[t] = (sums[t] / total[t, None]).astype(np.float32)
        colored[t] = True
        uncolored = int(np.count_nonzero(total[uncolored_list] <= 0))
        smooth_count += -1 if uncolored == last else 1
        last = uncolored
    return color, colored


def _forward(color, colored, src, dst, weight, nv):
    mask = colored.astype(np.float64)
    active = np.nonzero(colored)[0]
    color = color.astype(np.float64)
    while active.size:
        e = np.isin(src, active) & (mask[dst] <= 0)
        s, d, wgt = src[e], dst[e], weight[e]
        np.add.at(color, d, color[s] * wgt[:, None])
        np.add.at(mask, d, -wgt)
        pending = np.unique(d)
        color[pending] /= -mask[pending, None]
        mask[pending] = 1.0
        active = pending
    return color.astype(np.float32), mask == 1.0


def meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx, method="smooth"):
    h, w, c, corner_v, texel, seen, color, colored, src, dst, weight = _setup(
        texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx
    )
    nv = vtx_pos.shape[0]
    if method == "smooth":
        color, colored = _smooth(color, colored, corner_v, seen, src, dst, weight, nv)
    elif method == "forward":
        color, colored = _forward(color, colored, src, dst, weight, nv)
    else:
        raise ValueError("Invalid method. Use 'smooth' or 'forward'.")
    new_texture, new_mask = _write_back(texture, mask, color, colored, corner_v, texel, c)
    return new_texture.reshape(h, w, 3), new_mask.reshape(h, w)
