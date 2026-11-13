"""
Attention helpers for the multiview paint UNet.

Its multiview reference attention packs several materials into the values (value head dim = 2 x query head dim).
Fused attention kernels need equal head dims, so torch falls back to the math path and materializes the whole
(batch, heads, q_len, kv_len) score matrix: 5.6 GiB per call at 512 px with 6 views, a 21 GiB peak. Attention
weights do not depend on the values, so attending each value slice separately and concatenating is exact.

The UNet's code is loaded from the model snapshot (diffusers copies it into its dynamic-module cache), not from this
repo, so the fix is applied to the loaded module after `from_pretrained`.
"""

import sys

import torch.nn.functional as F


def split_value_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    d = query.shape[-1]
    if value.shape[-1] != d and value.shape[-1] % d == 0:
        import torch

        return torch.cat(
            [
                F.scaled_dot_product_attention(query, key, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                               is_causal=is_causal, **kwargs)
                for v in value.split(d, dim=-1)
            ],
            dim=-1,
        )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                          is_causal=is_causal, **kwargs)


class _FunctionalWithSplitValueSDPA:
    """torch.nn.functional, except scaled_dot_product_attention handles values wider than the queries."""

    scaled_dot_product_attention = staticmethod(split_value_sdpa)

    def __getattr__(self, name):
        return getattr(F, name)


def patch_unet_attention(unet) -> bool:
    """Route the loaded paint UNet's attention processors through split_value_sdpa. Returns True if patched."""
    package = type(unet).__module__.rpartition(".")[0]
    patched = False
    for name in (f"{package}.attn_processor", f"{package}.unet.attn_processor"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "F", None) is F:
            module.F = _FunctionalWithSplitValueSDPA()
            patched = True
    return patched
