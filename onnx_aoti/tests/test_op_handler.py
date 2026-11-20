import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from onnx_aoti import AOTIModule, compile_onnx_to_pt2  # noqa: E402
from onnx_op_handler import (  # noqa: E402
    CUSTOM_CONVERTERS,
    ONNXConversionError,
    convert_checked,
    register_converter,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def make_model(nodes, inputs, outputs, initializers=(), domains=None):
    graph = oh.make_graph(
        nodes,
        "g",
        [oh.make_tensor_value_info(n, onnx.TensorProto.FLOAT, s) for n, s in inputs],
        [onnx.ValueInfoProto(name=o) for o in outputs],
        initializer=list(initializers),
    )
    opsets = [oh.make_opsetid("", 21)] + [oh.make_opsetid(d, 1) for d in (domains or [])]
    model = oh.make_model(graph, opset_imports=opsets)
    model.ir_version = 10
    return onnx.shape_inference.infer_shapes(model)


def test_unknown_ops_reported_together():
    model = make_model(
        [
            oh.make_node("FooOp", ["x"], ["a"], name="foo"),
            oh.make_node("BarOp", ["a"], ["y"], name="bar"),
        ],
        inputs=[("x", [2, 3])],
        outputs=["y"],
    )
    try:
        convert_checked(model)
    except ONNXConversionError as e:
        nodes = sorted(p.node for p in e.problems)
        assert nodes == ["bar", "foo"], nodes
        assert all(p.level == "error" for p in e.problems)
    else:
        raise AssertionError("expected ONNXConversionError")
    print("[ok] unknown ops are reported together")


def test_custom_domain_rejected():
    model = make_model(
        [oh.make_node("Gelu", ["x"], ["y"], name="mse", domain="com.microsoft")],
        inputs=[("x", [2, 3])],
        outputs=["y"],
        domains=["com.microsoft"],
    )
    try:
        convert_checked(model)
    except ONNXConversionError as e:
        assert any("custom domain" in p.message for p in e.problems), e.problems
    else:
        raise AssertionError("expected ONNXConversionError")
    print("[ok] custom domain op is rejected")


def test_registered_converter_is_used():
    register_converter("FooOp", lambda node, c, opset: (lambda x, *_: x * 2))
    try:
        model = make_model(
            [oh.make_node("FooOp", ["x"], ["y"], name="foo")],
            inputs=[("x", [2, 3])],
            outputs=["y"],
        )
        torch_model = convert_checked(model)
        x = torch.randn(2, 3)
        with torch.no_grad():
            out = torch_model(x)
        assert torch.allclose(out, x * 2), "custom converter output mismatch"
    finally:
        CUSTOM_CONVERTERS.pop("FooOp", None)
    print("[ok] registered converter is used")


def test_data_dependent_warning():
    model = make_model(
        [oh.make_node("NonZero", ["x"], ["y"], name="nz")],
        inputs=[("x", [2, 3])],
        outputs=["y"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        convert_checked(model)
    messages = [str(w.message) for w in caught]
    assert any("NonZero" in m and "depends on the input values" in m for m in messages), messages
    print("[ok] data-dependent op emits a warning")


def test_aoti_end_to_end():
    # Reshape, Slice, ReduceMean, Unsqueeze: all need constant-folding to export.
    shape_c = nh.from_array(np.array([2, 12], np.int64), "shape")
    starts = nh.from_array(np.array([0], np.int64), "starts")
    ends = nh.from_array(np.array([1], np.int64), "ends")
    axes0 = nh.from_array(np.array([0], np.int64), "axes0")
    axes1 = nh.from_array(np.array([1], np.int64), "axes1")
    model = make_model(
        [
            oh.make_node("Reshape", ["x", "shape"], ["r"], name="reshape"),
            oh.make_node("Slice", ["r", "starts", "ends", "axes0"], ["s"], name="slice"),
            oh.make_node("ReduceMean", ["s", "axes1"], ["m"], name="mean", keepdims=0),
            oh.make_node("Unsqueeze", ["m", "axes0"], ["y"], name="unsqueeze"),
        ],
        inputs=[("x", [1, 4, 6])],
        outputs=["y"],
        initializers=[shape_c, starts, ends, axes0, axes1],
    )
    onnx_path = ARTIFACTS / "handler_net.onnx"
    pt2_path = ARTIFACTS / "handler_net.pt2"
    onnx.save(model, str(onnx_path))

    x = np.random.default_rng(0).standard_normal((1, 4, 6)).astype(np.float32)
    ref = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]

    compile_onnx_to_pt2(onnx_path, pt2_path, input_shape=[1, 4, 6])
    out = AOTIModule(pt2_path)(torch.from_numpy(x)).numpy()

    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert np.allclose(out, ref, atol=1e-5), float(np.abs(out - ref).max())
    print("[ok] AOTI compile + run matches onnxruntime")


if __name__ == "__main__":
    test_unknown_ops_reported_together()
    test_custom_domain_rejected()
    test_registered_converter_is_used()
    test_data_dependent_warning()
    test_aoti_end_to_end()
