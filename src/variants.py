"""Every inference backend exposes the same interface: f(X[B,30,24] float32) -> ndarray[B]."""
import os, platform
import numpy as np

NAMES = ["torch_fp32", "torch_int8", "ort_fp32", "ort_int8"]


def load(name, art="artifacts", threads=1):
    if name == "dummy":  # only for testing the real-time loop without any ML library
        w = np.random.RandomState(0).randn(720, 64).astype(np.float32)
        def f(X):
            h = np.tanh(X.reshape(len(X), -1) @ w)
            for _ in range(30):
                h = np.tanh(h * 0.99 + 0.01)
            return h.sum(1)
        return f
    if name.startswith("torch"):
        import torch, torch.nn as nn
        from src.model import LSTMRegressor
        torch.set_num_threads(threads)
        m = LSTMRegressor(); m.load_state_dict(torch.load(f"{art}/lstm_fp32.pt")); m.eval()
        if name == "torch_int8":
            torch.backends.quantized.engine = "qnnpack" if platform.machine() in ("aarch64", "arm64") else "fbgemm"
            m = torch.ao.quantization.quantize_dynamic(m, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
        def f(X):
            with torch.inference_mode():
                return m(torch.from_numpy(X)).numpy().ravel()
        return f
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(f"{art}/lstm_{name.split('_')[1]}.onnx", so, providers=["CPUExecutionProvider"])
    return lambda X: s.run(None, {"x": X})[0].ravel()
