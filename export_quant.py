"""Export fp32 -> ONNX, then ONNX Runtime dynamic INT8 quantization."""
import argparse, os, torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.model import LSTMRegressor

ap = argparse.ArgumentParser(); ap.add_argument("--art", default="artifacts"); a = ap.parse_args()
m = LSTMRegressor(); m.load_state_dict(torch.load(f"{a.art}/lstm_fp32.pt")); m.eval()
x = torch.zeros(1, 30, 24)
fp32 = f"{a.art}/lstm_fp32.onnx"
kw = dict(input_names=["x"], output_names=["rul"], dynamic_axes={"x": {0: "b"}, "rul": {0: "b"}}, opset_version=17)
try:
    torch.onnx.export(m, x, fp32, dynamo=False, **kw)   # legacy exporter keeps a plain ONNX LSTM op
except TypeError:
    torch.onnx.export(m, x, fp32, **kw)
quantize_dynamic(fp32, f"{a.art}/lstm_int8.onnx", weight_type=QuantType.QInt8)
for f in ("lstm_fp32.pt", "lstm_fp32.onnx", "lstm_int8.onnx"):
    print(f"{f:18s} {os.path.getsize(f'{a.art}/{f}')/1024:8.1f} KB")
