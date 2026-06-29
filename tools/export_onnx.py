import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import sys
import os

sys.path.append(r"C:\Users\Administrator\Desktop\Training")
from checkpoint_utils import load_model_from_checkpoint

def main():
    model_path = r"C:\Users\Administrator\Downloads\model_latest.pth"
    device = "cpu"
    print("1. Loading PyTorch model...")
    net = load_model_from_checkpoint(model_path, device, model_class="v8_legacy", board_size=15)
    net.eval()

    # Determine input channels (usually 4 for our MCTS implementation)
    in_channels = 6
    dummy_input = torch.zeros(64, in_channels, 15, 15, dtype=torch.float32)

    fp32_onnx_path = r"C:\Users\Administrator\Desktop\Training\model_latest_fp32.onnx"
    int8_onnx_path = r"C:\Users\Administrator\Desktop\Training\model_latest_int8.onnx"

    print(f"2. Exporting to FP32 ONNX: {fp32_onnx_path}")
    torch.onnx.export(
        net,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"]
    )

    print(f"3. Preprocessing ONNX model for quantization...")
    preprocessed_path = fp32_onnx_path.replace(".onnx", "_prep.onnx")
    quant_pre_process(fp32_onnx_path, preprocessed_path, skip_onnx_shape=True)

    print(f"4. Quantizing to INT8 ONNX: {int8_onnx_path}")
    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QUInt8
    )

    print("\n[SUCCESS] Export Completed!")
    print(f"FP32 Size: {os.path.getsize(fp32_onnx_path) / (1024*1024):.2f} MB")
    print(f"INT8 Size: {os.path.getsize(int8_onnx_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()
