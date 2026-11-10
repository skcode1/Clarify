"""Export PyTorch EDSR model from super-image to an ONNX graph with dynamic axes."""

from pathlib import Path
from pyexpat import model
import torch
from super_image import EdsrModel

def export_edsr_to_onnx(scale: int = 2, output_path: str = "src/clarify/models/edsr_x2.onnx") -> None:
    print(f"Loading pretrained EDSR (scale={scale})...")
    model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)
    model.eval()

  
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
   
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    }

    print("Exporting to ONNX with dynamic input/output shapes...")
    torch.onnx.export(
        raw_model,
        dummy_input,
        str(dest),
        export_params=True,
        opset_version=14,          
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,              
    )
    
    print(f"ONNX export successful: {dest} ({dest.stat().st_size / (1024 * 1024):.2f} MB)")

if __name__ == "__main__":
    export_edsr_to_onnx()
