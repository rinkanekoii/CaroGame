import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
sys.path.append(r'C:\Users\Administrator\Desktop\Training')
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from checkpoint_utils import load_model_from_checkpoint

print("Loading model...")
net = load_model_from_checkpoint(r'C:\Users\Administrator\Downloads\model_latest.pth', 'cpu', model_class='v8_legacy', board_size=15)
net.eval()
dummy_input = torch.zeros(1, 6, 15, 15)

model_fp32 = r'C:\Users\Administrator\Desktop\CaroNet\backend\model_latest_fp32.onnx'
model_int8 = r'C:\Users\Administrator\Desktop\CaroNet\backend\model_latest_int8.onnx'

print("Exporting to FP32 ONNX...")
torch.onnx.export(
    net, dummy_input, model_fp32, export_params=True, opset_version=17, 
    input_names=['input'], output_names=['policy', 'value'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'policy': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
)

print("Clearing value_info to bypass shape inference bugs...")
model = onnx.load(model_fp32)
model.graph.value_info.clear()
onnx.save(model, model_fp32)

print("Quantizing to INT8 ONNX...")
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QUInt8)

print("INT8 model exported successfully to", model_int8)
