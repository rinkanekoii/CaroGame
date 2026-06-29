import torch
import onnx
import sys
import os

sys.path.append(r"C:\Users\Administrator\Desktop\Training")
from checkpoint_utils import load_model_from_checkpoint

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    model_path = r"C:\Users\Administrator\Downloads\model_latest.pth"
    device = "cpu"
    print("1. Loading PyTorch model...")
    net = load_model_from_checkpoint(model_path, device, model_class="v8_legacy", board_size=15)
    net.eval()

    in_channels = 6
    dummy_input = torch.zeros(2, in_channels, 15, 15, dtype=torch.float32)

    fp32_onnx_path = r"C:\Users\Administrator\Desktop\CaroNet\backend\model_latest_fp32.onnx"

    print(f"2. Exporting to FP32 ONNX: {fp32_onnx_path}")
    torch.onnx.export(
        net,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={"input": {0: "batch_size"}, "policy": {0: "batch_size"}, "value": {0: "batch_size"}}
    )

    print("\n[SUCCESS] Export Completed!")

if __name__ == "__main__":
    main()
