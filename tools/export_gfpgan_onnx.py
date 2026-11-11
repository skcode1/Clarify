"""Offline exporter: PyTorch GFPGAN -> ONNX (opset 14).

Run this in your development environment (with PyTorch and GFPGAN installed)
to generate the face restoration ONNX artifact consumed by clarify at runtime.
PyTorch is NOT a runtime dependency of clarify-ai; it is only used here offline.
"""

from __future__ import annotations

from pathlib import Path
import torch


def export_gfpgan(
    checkpoint: str = "GFPGANv1.4.pth",
    output_path: str | Path = "src/clarify/models/gfpgan_v1_4.onnx",
    out_size: int = 512,
    opset_version: int = 14,
) -> Path:
    """Export the GFPGANv1Clean architecture to ONNX."""
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

    model = GFPGANv1Clean(
        out_size=out_size,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["params_ema"])
    model.eval()

    dummy = torch.randn(1, 3, out_size, out_size, dtype=torch.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_face"],
        output_names=["restored_face"],
    )
    print(f"Exported: {output}")
    return output


if __name__ == "__main__":
    export_gfpgan()
