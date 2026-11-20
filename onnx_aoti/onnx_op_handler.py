"""Checked ONNX -> PyTorch conversion.

onnx2pytorch passes constant shape/axes/index inputs as buffers and reads them
with .item(), which makes torch.export fail with data-dependent errors. The
converters below bake those constants into the module as Python values, so the
exported graph is static. Nodes that fail to convert are collected and reported
together instead of stopping at the first error.
"""

import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch
import torch.nn.functional as F
from torch import nn

import onnx2pytorch.convert.model as _model_mod
from onnx2pytorch.convert.operations import convert_operations as _original_convert_operations
from onnx2pytorch.convert.model import ConvertModel

DEFAULT_DOMAINS = ("", "ai.onnx")

DATA_DEPENDENT_OPS = {
    "NonZero": "output shape depends on the input values",
    "NonMaxSuppression": "output shape depends on the input values",
    "Unique": "output shape depends on the input values",
    "Loop": "control flow with data-dependent trip count",
    "If": "control flow with data-dependent branch",
    "Scan": "control flow",
    "SequenceMap": "control flow",
}

ABSENT = object()


class ONNXConversionError(RuntimeError):
    def __init__(self, problems):
        self.problems = problems
        lines = [str(p) for p in problems]
        super().__init__("ONNX conversion failed:\n  " + "\n  ".join(lines))


@dataclass
class Issue:
    level: str  # "error" or "warning"
    node: str
    op_type: str
    message: str

    def __str__(self):
        where = f"node '{self.node}' ({self.op_type})" if self.op_type else "model"
        return f"[{self.level}] {where}: {self.message}"


class _Consts:
    """Constant values visible to a node: initializers and Constant node outputs."""

    def __init__(self, graph):
        self._tensors = {t.name: t for t in graph.initializer}
        self._values = {}
        for node in graph.node:
            if node.op_type == "Constant" and node.output:
                self._values[node.output[0]] = _constant_node_value(node)

    def get(self, name):
        if name in self._values:
            return self._values[name]
        if name in self._tensors:
            self._values[name] = numpy_helper.to_array(self._tensors.pop(name))
            return self._values[name]
        return None


def _constant_node_value(node):
    for attr in node.attribute:
        if attr.name == "value":
            return numpy_helper.to_array(attr.t)
        if attr.name in ("value_float", "value_int", "value_floats", "value_ints"):
            return np.asarray(helper.get_attribute_value(attr))
    raise NotImplementedError(f"Constant node '{node.name}' has an unsupported attribute")


def _attr(node, name):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return None


def _attr_int(node, name, default):
    value = _attr(node, name)
    return default if value is None else int(value)


def _attr_str(node, name, default):
    value = _attr(node, name)
    if value is None:
        return default
    return value.decode() if isinstance(value, bytes) else str(value)


def _opt_const(node, c, idx, attr=None):
    """Constant value of an optional input. None if dynamic, ABSENT if not given."""
    if idx < len(node.input) and node.input[idx]:
        return c.get(node.input[idx])
    if attr is not None:
        value = _attr(node, attr)
        if value is not None:
            return np.asarray(value)
    return ABSENT


def _required(node, c, idx):
    if idx >= len(node.input) or not node.input[idx]:
        raise ValueError(f"{node.op_type} requires input {idx}")
    return c.get(node.input[idx])


def _ints(value):
    return [int(v) for v in np.asarray(value).reshape(-1)]


def _reshape(node, c, opset):
    shape = _required(node, c, 1)
    if shape is None:
        return None
    dims = _ints(shape)
    allowzero = _attr_int(node, "allowzero", 0)

    def fn(x, *_):
        target = [x.shape[i] if (s == 0 and not allowzero) else s for i, s in enumerate(dims)]
        return x.reshape(target)

    return fn


def _unsqueeze(node, c, opset):
    axes = _opt_const(node, c, 1, attr="axes")
    if axes is None:
        return None
    if axes is ABSENT:
        raise ValueError("Unsqueeze requires axes")
    axes = _ints(axes)

    def fn(x, *_):
        rank = x.dim() + len(axes)
        y = x
        for a in sorted(a % rank for a in axes):
            y = y.unsqueeze(a)
        return y

    return fn


def _squeeze(node, c, opset):
    axes = _opt_const(node, c, 1, attr="axes")
    if axes is None:
        return None
    if axes is ABSENT:
        return lambda x, *_: x.squeeze()
    dims = tuple(_ints(axes))
    return lambda x, *_: x.squeeze(dims)


def _prod(x, dims, keep):
    y = x
    for d in sorted(dims, reverse=True):
        y = torch.prod(y, d, keepdim=True)
    return y if keep else y.squeeze(dims)


_REDUCERS = {
    "ReduceMean": lambda x, d, k: torch.mean(x, d, keepdim=k),
    "ReduceSum": lambda x, d, k: torch.sum(x, d, keepdim=k),
    "ReduceMax": lambda x, d, k: torch.amax(x, d, keepdim=k),
    "ReduceMin": lambda x, d, k: torch.amin(x, d, keepdim=k),
    "ReduceProd": _prod,
    "ReduceL1": lambda x, d, k: torch.sum(torch.abs(x), d, keepdim=k),
    "ReduceL2": lambda x, d, k: torch.sqrt(torch.sum(x * x, d, keepdim=k)),
    "ReduceSumSquare": lambda x, d, k: torch.sum(x * x, d, keepdim=k),
    "ReduceLogSumExp": lambda x, d, k: torch.logsumexp(x, d, keepdim=k),
}


def _make_reduce(reducer):
    def convert(node, c, opset):
        axes = _opt_const(node, c, 1, attr="axes")
        if axes is None:
            return None
        keep = bool(_attr_int(node, "keepdims", 1))
        noop = bool(_attr_int(node, "noop_with_empty_axes", 0))
        axes = None if axes is ABSENT or len(np.asarray(axes).reshape(-1)) == 0 else _ints(axes)

        def fn(x, *_):
            if axes is None and noop:
                return x
            dims = tuple(range(x.dim())) if axes is None else tuple(a % x.dim() for a in axes)
            return reducer(x, dims, keep)

        return fn

    return convert


def _topk(node, c, opset):
    k = _required(node, c, 1)
    if k is None:
        return None
    k = _ints(k)[0]
    axis = _attr_int(node, "axis", -1)
    largest = bool(_attr_int(node, "largest", 1))
    sorted_ = bool(_attr_int(node, "sorted", 1))

    def fn(x, *_):
        return torch.topk(x, k, dim=axis, largest=largest, sorted=sorted_)

    return fn


def _cumsum(node, c, opset):
    if _attr_int(node, "exclusive", 0) or _attr_int(node, "reverse", 0):
        raise NotImplementedError("CumSum with exclusive or reverse is not supported")
    axis = _required(node, c, 1)
    if axis is None:
        return None
    axis = _ints(axis)[0]
    return lambda x, *_: torch.cumsum(x, dim=axis)


def _slice_for(start, end, step, dim):
    if step > 0:
        return slice(start, end, step)
    # Negative step: ONNX clamps differently from Python slicing, so normalize here.
    if start < 0:
        start += dim
    start = min(max(start, -1), dim - 1)
    if start < 0:
        return slice(0, 0)
    if end < 0:
        end += dim
        if end < 0:
            end = None
    return slice(start, end, step)


def _slice(node, c, opset):
    starts = _required(node, c, 1)
    ends = _required(node, c, 2)
    if starts is None or ends is None:
        return None
    axes = _opt_const(node, c, 3)
    steps = _opt_const(node, c, 4)
    if axes is None or steps is None:
        return None
    starts, ends = _ints(starts), _ints(ends)
    axes = list(range(len(starts))) if axes is ABSENT else _ints(axes)
    steps = [1] * len(starts) if steps is ABSENT else _ints(steps)

    def fn(x, *_):
        index = [slice(None)] * x.dim()
        for s, e, a, st in zip(starts, ends, axes, steps):
            d = a % x.dim()
            index[d] = _slice_for(s, e, st, x.shape[d])
        return x[tuple(index)]

    return fn


def _expand(node, c, opset):
    shape = _required(node, c, 1)
    if shape is None:
        return None
    dims = tuple(_ints(shape))

    def fn(x, *_):
        return x.expand(torch.broadcast_shapes(tuple(x.shape), dims))

    return fn


def _pad(node, c, opset):
    pads = _required(node, c, 1)
    if pads is None:
        return None
    if len(node.input) > 3 and node.input[3]:
        raise NotImplementedError("Pad with the axes input is not supported")
    value = 0.0
    if len(node.input) > 2 and node.input[2]:
        constant = c.get(node.input[2])
        if constant is None:
            return None
        value = float(np.asarray(constant).reshape(-1)[0])
    mode = _attr_str(node, "mode", "constant")
    mode_map = {"constant": "constant", "reflect": "reflect", "edge": "replicate"}
    if mode not in mode_map:
        raise NotImplementedError(f"Pad mode '{mode}' is not supported")
    p = _ints(pads)
    rank = len(p) // 2
    pad_list = []
    for d in reversed(range(rank)):
        pad_list += [p[d], p[d + rank]]

    def fn(x, *_):
        if mode == "constant":
            return F.pad(x, pad_list, mode="constant", value=value)
        return F.pad(x, pad_list, mode=mode_map[mode])

    return fn


def _constant_of_shape(node, c, opset):
    shape = _required(node, c, 0)
    if shape is None:
        return None
    dims = tuple(_ints(shape))
    value = _attr(node, "value")
    arr = numpy_helper.to_array(value) if value is not None else np.zeros(1, np.float32)
    scalar = arr.reshape(-1)[0].item()
    dtype = torch.from_numpy(np.ascontiguousarray(arr)).dtype

    def fn(*_):
        return torch.full(dims, scalar, dtype=dtype)

    return fn


def _resize(node, c, opset):
    scales = _opt_const(node, c, 2)
    sizes = _opt_const(node, c, 3)
    if scales is None or sizes is None:
        return None
    if scales is ABSENT and sizes is ABSENT:
        raise ValueError("Resize requires scales or sizes")
    mode = _attr_str(node, "mode", "nearest")
    coord = _attr_str(node, "coordinate_transformation_mode", "half_pixel")
    if mode == "cubic":
        raise NotImplementedError("Resize mode 'cubic' is not supported")
    if coord not in ("half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric"):
        raise NotImplementedError(f"Resize coordinate_transformation_mode '{coord}' is not supported")
    align = coord == "align_corners"
    if mode == "nearest" and align:
        raise NotImplementedError("Resize nearest with align_corners is not supported")
    torch_mode = "nearest" if mode == "nearest" else {1: "linear", 2: "bilinear", 3: "trilinear"}

    if sizes is not ABSENT:
        spatial = tuple(_ints(sizes)[2:])

        def fn(x, *_):
            mode_ = torch_mode if isinstance(torch_mode, str) else torch_mode[x.dim() - 2]
            return F.interpolate(x, size=spatial, mode=mode_, align_corners=None if mode_ == "nearest" else align)

        return fn

    factors = tuple(float(s) for s in np.asarray(scales).reshape(-1)[2:])
    if mode == "nearest" and any(f < 1 or not f.is_integer() for f in factors):
        raise NotImplementedError("Resize nearest is only exact for integer upscaling")

    def fn(x, *_):
        mode_ = torch_mode if isinstance(torch_mode, str) else torch_mode[x.dim() - 2]
        return F.interpolate(
            x,
            scale_factor=factors,
            mode=mode_,
            align_corners=None if mode_ == "nearest" else align,
            recompute_scale_factor=True,
        )

    return fn


def _onehot(node, c, opset):
    depth = _required(node, c, 1)
    values = _required(node, c, 2)
    if depth is None or values is None:
        return None
    depth = _ints(depth)[0]
    values = np.asarray(values).reshape(-1)
    off, on = values[0].item(), values[1].item()
    dtype = torch.from_numpy(np.ascontiguousarray(values)).dtype
    axis = _attr_int(node, "axis", -1)

    def fn(indices, *_):
        hot = F.one_hot(indices.long(), depth).bool()
        out = torch.where(
            hot,
            torch.tensor(on, dtype=dtype),
            torch.tensor(off, dtype=dtype),
        )
        return out.movedim(-1, axis % (indices.dim() + 1))

    return fn


CUSTOM_CONVERTERS = {
    "Reshape": _reshape,
    "Unsqueeze": _unsqueeze,
    "Squeeze": _squeeze,
    "TopK": _topk,
    "CumSum": _cumsum,
    "Slice": _slice,
    "Expand": _expand,
    "Pad": _pad,
    "ConstantOfShape": _constant_of_shape,
    "Resize": _resize,
    "OneHot": _onehot,
    **{op: _make_reduce(fn) for op, fn in _REDUCERS.items()},
}


def register_converter(op_type: str, converter: Callable) -> None:
    """Register converter(node, consts, opset) -> fn(*inputs) or None to defer to onnx2pytorch."""
    CUSTOM_CONVERTERS[op_type] = converter


class _ConstOp(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class _ConstMultiOp(_ConstOp):
    pass


# onnx2pytorch routes multi-output modules through a separate branch; register ours there.
if _ConstMultiOp not in _model_mod.MULTIOUTPUT_LAYERS:
    _model_mod.MULTIOUTPUT_LAYERS = tuple(_model_mod.MULTIOUTPUT_LAYERS) + (_ConstMultiOp,)


def _single_node_graph(node, initializers):
    graph = onnx.GraphProto()
    graph.node.append(node)
    for name in node.input:
        if name in initializers:
            graph.initializer.add().CopyFrom(initializers[name])
    return graph


def _convert_node(node, consts, initializers, opset_version, batch_dim, enable_pruning):
    custom = CUSTOM_CONVERTERS.get(node.op_type)
    if custom is not None:
        fn = custom(node, consts, opset_version)
        if fn is not None:
            cls = _ConstMultiOp if len(node.output) > 1 else _ConstOp
            return [(node.output[0], f"{node.op_type}_{node.output[0]}", cls(fn))]
    graph = _single_node_graph(node, initializers)
    return list(_original_convert_operations(graph, opset_version, batch_dim, enable_pruning))


def _convert_operations_checked(onnx_graph, opset_version, batch_dim=0, enable_pruning=True):
    consts = _Consts(onnx_graph)
    initializers = {t.name: t for t in onnx_graph.initializer}
    problems = []
    for node in onnx_graph.node:
        try:
            converted = _convert_node(node, consts, initializers, opset_version, batch_dim, enable_pruning)
        except Exception as e:
            problems.append(Issue("error", node.name or node.output[0], node.op_type, f"{type(e).__name__}: {e}"))
            continue
        yield from converted
    if problems:
        raise ONNXConversionError(problems)


@contextmanager
def _checked_conversion():
    original = _model_mod.convert_operations
    _model_mod.convert_operations = _convert_operations_checked
    try:
        yield
    finally:
        _model_mod.convert_operations = original


def preflight(onnx_model: onnx.ModelProto) -> list:
    issues = []

    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        issues.append(Issue("warning", "", "", f"ONNX checker failed: {e}"))

    graph = onnx_model.graph
    initializer_names = {t.name for t in graph.initializer}

    for node in graph.node:
        name = node.name or node.output[0]
        if node.domain not in DEFAULT_DOMAINS:
            issues.append(Issue("error", name, node.op_type, f"custom domain '{node.domain}' has no converter"))
        if node.op_type in DATA_DEPENDENT_OPS:
            issues.append(Issue("warning", name, node.op_type, DATA_DEPENDENT_OPS[node.op_type]))
        if node.op_type == "Compress" and node.input[1] not in initializer_names:
            issues.append(Issue("warning", name, node.op_type, "condition is not constant, output shape is data-dependent"))

    for vi in graph.input:
        if vi.name in initializer_names:
            continue
        for dim in vi.type.tensor_type.shape.dim:
            if dim.dim_value <= 0:
                issues.append(Issue("warning", vi.name, "", "dynamic input dimension, the compiled package is static"))
                break

    return issues


def convert_checked(onnx_model: onnx.ModelProto) -> nn.Module:
    """Convert an ONNX model to PyTorch, reporting every problem found.

    Raises ONNXConversionError listing all nodes that could not be converted.
    Warnings from preflight are emitted with warnings.warn.
    """
    issues = preflight(onnx_model)
    for issue in issues:
        if issue.level == "warning":
            warnings.warn(str(issue), stacklevel=2)
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        raise ONNXConversionError(errors)

    with _checked_conversion():
        return ConvertModel(onnx_model).eval()
