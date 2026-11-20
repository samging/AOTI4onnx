"""Sweep common ONNX ops through ConvertModel -> torch.export and compare with onnxruntime.

Each spec is a single-node ONNX model. Graph input is always "x"; every other
name in `ins` must be a constant listed in `consts`.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
import onnxruntime as ort
import torch
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from onnx_op_handler import ONNXConversionError, convert_checked  # noqa: E402

OPSET = 21
rng = np.random.default_rng(0)


def f(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def pos(*shape):
    return (np.abs(rng.standard_normal(shape)) + 1.0).astype(np.float32)


I64 = np.int64
X4 = (1, 3, 8, 8)

SPECS = [
    # name, op, inputs, n_outputs, attrs, consts, x shape
    ("Relu", "Relu", ["x"], 1, {}, {}, X4),
    ("Sigmoid", "Sigmoid", ["x"], 1, {}, {}, X4),
    ("Tanh", "Tanh", ["x"], 1, {}, {}, X4),
    ("Exp", "Exp", ["x"], 1, {}, {}, X4),
    ("Neg", "Neg", ["x"], 1, {}, {}, X4),
    ("Erf", "Erf", ["x"], 1, {}, {}, X4),
    ("Abs", "Abs", ["x"], 1, {}, {}, X4),
    ("Floor", "Floor", ["x"], 1, {}, {}, X4),
    ("Ceil", "Ceil", ["x"], 1, {}, {}, X4),
    ("Sign", "Sign", ["x"], 1, {}, {}, X4),
    ("Round", "Round", ["x"], 1, {}, {}, X4),
    ("Softplus", "Softplus", ["x"], 1, {}, {}, X4),
    ("Elu", "Elu", ["x"], 1, {}, {}, X4),
    ("Selu", "Selu", ["x"], 1, {}, {}, X4),
    ("HardSwish", "HardSwish", ["x"], 1, {}, {}, X4),
    ("HardSigmoid", "HardSigmoid", ["x"], 1, {}, {}, X4),
    ("Mish", "Mish", ["x"], 1, {}, {}, X4),
    ("Gelu", "Gelu", ["x"], 1, {"approximate": "tanh"}, {}, X4),
    ("LeakyRelu", "LeakyRelu", ["x"], 1, {"alpha": 0.1}, {}, X4),
    ("Softmax", "Softmax", ["x"], 1, {"axis": -1}, {}, X4),
    ("LogSoftmax", "LogSoftmax", ["x"], 1, {"axis": 1}, {}, X4),
    ("Identity", "Identity", ["x"], 1, {}, {}, X4),
    ("Dropout", "Dropout", ["x"], 1, {}, {}, X4),
    ("Add", "Add", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Sub", "Sub", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Mul", "Mul", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Div", "Div", ["x", "c"], 1, {}, {"c": pos(1, 3, 1, 1)}, X4),
    ("Pow", "Pow", ["x", "c"], 1, {}, {"c": np.array(2.0, np.float32)}, X4),
    ("Max", "Max", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Min", "Min", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Mod", "Mod", ["x", "c"], 1, {"fmod": 1}, {"c": pos(1, 3, 1, 1)}, X4),
    ("Sum", "Sum", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Mean", "Mean", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Equal", "Equal", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Less", "Less", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Greater", "Greater", ["x", "c"], 1, {}, {"c": f(1, 3, 1, 1)}, X4),
    ("Where", "Where", ["cond", "x", "c"], 1, {}, {"cond": rng.random(X4) > 0.5, "c": f(1, 3, 1, 1)}, X4),
    ("Clip", "Clip", ["x", "lo", "hi"], 1, {}, {"lo": np.array(-0.5, np.float32), "hi": np.array(0.5, np.float32)}, X4),
    ("ReduceMean", "ReduceMean", ["x", "ax"], 1, {"keepdims": 1}, {"ax": np.array([2, 3], I64)}, X4),
    ("ReduceSum", "ReduceSum", ["x", "ax"], 1, {"keepdims": 1}, {"ax": np.array([2, 3], I64)}, X4),
    ("ReduceMax", "ReduceMax", ["x", "ax"], 1, {"keepdims": 0}, {"ax": np.array([2, 3], I64)}, X4),
    ("ReduceMin", "ReduceMin", ["x", "ax"], 1, {"keepdims": 0}, {"ax": np.array([2, 3], I64)}, X4),
    ("ReduceProd", "ReduceProd", ["x", "ax"], 1, {"keepdims": 0}, {"ax": np.array([2], I64)}, X4),
    ("ReduceL2", "ReduceL2", ["x", "ax"], 1, {"keepdims": 0}, {"ax": np.array([2, 3], I64)}, X4),
    ("ReduceLogSumExp", "ReduceLogSumExp", ["x", "ax"], 1, {"keepdims": 0}, {"ax": np.array([2, 3], I64)}, X4),
    ("ArgMax", "ArgMax", ["x"], 1, {"axis": 1, "keepdims": 0}, {}, X4),
    ("ArgMin", "ArgMin", ["x"], 1, {"axis": 1, "keepdims": 0}, {}, X4),
    ("TopK", "TopK", ["x", "k"], 2, {"axis": -1}, {"k": np.array([3], I64)}, X4),
    ("CumSum", "CumSum", ["x", "ax"], 1, {}, {"ax": np.array(2, I64)}, X4),
    ("Reshape", "Reshape", ["x", "shape"], 1, {}, {"shape": np.array([1, -1], I64)}, X4),
    ("Transpose", "Transpose", ["x"], 1, {"perm": [0, 2, 3, 1]}, {}, X4),
    ("Flatten", "Flatten", ["x"], 1, {"axis": 1}, {}, X4),
    ("Unsqueeze", "Unsqueeze", ["x", "ax"], 1, {}, {"ax": np.array([0], I64)}, X4),
    ("Squeeze", "Squeeze", ["x", "ax"], 1, {}, {"ax": np.array([0], I64)}, (1, 3, 8, 8)),
    ("Concat", "Concat", ["x", "x"], 1, {"axis": 1}, {}, X4),
    ("Split", "Split", ["x", "sp"], 2, {"axis": 1}, {"sp": np.array([1, 2], I64)}, X4),
    ("Slice", "Slice", ["x", "st", "en", "ax", "sp"],
     1, {}, {"st": np.array([0], I64), "en": np.array([2], I64), "ax": np.array([1], I64), "sp": np.array([1], I64)}, X4),
    ("Gather", "Gather", ["x", "idx"], 1, {"axis": 2}, {"idx": np.array([0, 2, 4], I64)}, X4),
    ("Expand", "Expand", ["x", "shape"], 1, {}, {"shape": np.array([2, 3, 8, 8], I64)}, X4),
    ("Tile", "Tile", ["x", "rep"], 1, {}, {"rep": np.array([1, 1, 2, 1], I64)}, X4),
    ("Pad", "Pad", ["x", "pads"], 1, {"mode": "constant"}, {"pads": np.array([0, 0, 1, 1, 0, 0, 1, 1], I64)}, X4),
    ("Cast", "Cast", ["x"], 1, {"to": 1}, {}, X4),
    ("Shape", "Shape", ["x"], 1, {}, {}, X4),
    ("ConstantOfShape", "ConstantOfShape", ["shape"], 1, {}, {"shape": np.array([1, 3], I64)}, X4),
    ("Resize", "Resize", ["x", "", "scales"], 1, {"mode": "nearest"},
     {"scales": np.array([1, 1, 2, 2], np.float32)}, X4),
    ("SpaceToDepth", "SpaceToDepth", ["x"], 1, {"blocksize": 2}, {}, X4),
    ("DepthToSpace", "DepthToSpace", ["x"], 1, {"blocksize": 2}, {}, (1, 12, 4, 4)),
    ("Conv", "Conv", ["x", "w", "b"], 1, {"pads": [1, 1, 1, 1]}, {"w": f(8, 3, 3, 3), "b": f(8)}, X4),
    ("ConvTranspose", "ConvTranspose", ["x", "w"], 1, {"strides": [2, 2]}, {"w": f(3, 4, 2, 2)}, X4),
    ("MaxPool", "MaxPool", ["x"], 1, {"kernel_shape": [2, 2], "strides": [2, 2]}, {}, X4),
    ("AveragePool", "AveragePool", ["x"], 1, {"kernel_shape": [2, 2], "strides": [2, 2]}, {}, X4),
    ("LpPool", "LpPool", ["x"], 1, {"kernel_shape": [2, 2], "strides": [2, 2], "p": 2}, {}, X4),
    ("GlobalAveragePool", "GlobalAveragePool", ["x"], 1, {}, {}, X4),
    ("GlobalMaxPool", "GlobalMaxPool", ["x"], 1, {}, {}, X4),
    ("BatchNormalization", "BatchNormalization", ["x", "s", "b", "m", "v"], 1, {"epsilon": 1e-5},
     {"s": pos(3), "b": f(3), "m": f(3), "v": pos(3)}, X4),
    ("InstanceNormalization", "InstanceNormalization", ["x", "s", "b"], 1, {},
     {"s": pos(3), "b": f(3)}, X4),
    ("LayerNormalization", "LayerNormalization", ["x", "s", "b"], 1, {"axis": -1},
     {"s": pos(8), "b": f(8)}, X4),
    ("GroupNormalization", "GroupNormalization", ["x", "s", "b"], 1, {"num_groups": 1},
     {"s": pos(3), "b": f(3)}, X4),
    ("LRN", "LRN", ["x"], 1, {"size": 3}, {}, X4),
    ("MatMul", "MatMul", ["x", "w"], 1, {}, {"w": f(8, 4)}, X4),
    ("Gemm", "Gemm", ["x", "w", "b"], 1, {"transB": 1}, {"w": f(5, 16), "b": f(5)}, (2, 16)),
    ("Einsum", "Einsum", ["x", "x"], 1, {"equation": "bchw,bchw->bc"}, {}, X4),
    ("OneHot", "OneHot", ["idx", "depth", "vals"], 1, {"axis": -1},
     {"idx": np.array([1, 2], I64), "depth": np.array(4, I64), "vals": np.array([0.0, 1.0], np.float32)}, (2,)),
    ("Compress", "Compress", ["x", "cond"], 1, {"axis": 1},
     {"cond": np.array([True, False, True])}, X4),
    ("QuantizeLinear", "QuantizeLinear", ["x", "sc", "zp"], 1, {}, {"sc": np.array(0.1, np.float32), "zp": np.array(0, np.int8)}, X4),
    ("DequantizeLinear", "DequantizeLinear", ["q", "sc", "zp"], 1, {},
     {"q": rng.integers(-100, 100, X4).astype(np.int8), "sc": np.array(0.1, np.float32), "zp": np.array(0, np.int8)}, X4),
    ("NonZero", "NonZero", ["x"], 1, {}, {}, X4),
]


def build_model(spec):
    name, op, ins, n_out, attrs, consts, xshape = spec
    outs = [f"y{i}" for i in range(n_out)]
    node = oh.make_node(op, ins, outs, name=f"{op}_node", **attrs)
    inits = [nh.from_array(np.asarray(v), name=k) for k, v in consts.items()]
    graph = oh.make_graph(
        [node],
        f"g_{name}",
        [oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, list(xshape))],
        [onnx.ValueInfoProto(name=o) for o in outs],
        initializer=inits,
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", OPSET)])
    model.ir_version = 10
    return onnx.shape_inference.infer_shapes(model)


def as_list(out):
    if isinstance(out, torch.Tensor):
        return [out]
    return list(out)


def run_spec(spec):
    model = build_model(spec)
    x = f(*spec[6])
    ref = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})

    try:
        torch_model = convert_checked(model)
    except ONNXConversionError as e:
        return "convert_fail", "; ".join(str(p) for p in e.problems)[:200]
    except Exception as e:
        return "convert_fail", f"{type(e).__name__}: {e}"

    try:
        with torch.no_grad():
            out = as_list(torch_model(torch.from_numpy(x)))
    except Exception as e:
        return "run_fail", f"{type(e).__name__}: {e}"

    for r, o in zip(ref, out):
        o = o.numpy()
        if r.dtype == np.bool_ or np.issubdtype(r.dtype, np.integer):
            if not np.array_equal(r, o.astype(r.dtype)):
                return "mismatch", "integer/bool output differs"
        elif not np.allclose(r, o, rtol=1e-3, atol=1e-4, equal_nan=True):
            return "mismatch", f"max abs diff {np.nanmax(np.abs(r - o)):.2e}"

    try:
        with torch.no_grad():
            torch.export.export(torch_model, (torch.from_numpy(x),))
    except Exception as e:
        first = str(e).strip().splitlines()[0][:160]
        return "export_fail", f"{type(e).__name__}: {first}"

    return "ok", ""


def main():
    results = []
    for spec in SPECS:
        try:
            status, msg = run_spec(spec)
        except Exception as e:
            status, msg = "harness_error", f"{type(e).__name__}: {e}"
        results.append((spec[0], status, msg))

    width = max(len(r[0]) for r in results)
    for name, status, msg in results:
        print(f"{name:<{width}}  {status:<14} {msg}")

    failed = [r for r in results if r[1] != "ok"]
    print(f"\n{len(results) - len(failed)}/{len(results)} ok")


if __name__ == "__main__":
    main()
