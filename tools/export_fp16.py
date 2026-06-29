import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
sys.path.append(r"C:\Users\Administrator\Desktop\Training")
import torch
import os
from checkpoint_utils import load_model_from_checkpoint

def main():
    model_path = r"C:\Users\Administrator\Downloads\model_latest.pth"
    device = "cpu"
    print("Loading PyTorch model...")
    net = load_model_from_checkpoint(model_path, device, model_class="v8_legacy", board_size=15)
    net.eval()
    
    # Convert weights to FP16
    net.half()
    
    in_channels = 6
    # Create FP16 dummy input
    dummy_input = torch.zeros(1, in_channels, 15, 15, dtype=torch.float16)
    
    out_path = r"C:\Users\Administrator\Desktop\CaroNet\backend\model_latest_fp16.onnx"
    print(f"Exporting to FP16 ONNX: {out_path}")
    
    # Use opset 17 or 18 (PyTorch might warn but it will export)
    torch.onnx.export(
        net,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"}
        }
    )
    
    print(f"SUCCESS! FP16 model saved to {out_path}")
    print(f"Size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
