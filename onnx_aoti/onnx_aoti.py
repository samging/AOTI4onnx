# onnx_aoti.py
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from onnx_op_handler import convert_checked

META_NAME = "model_meta.json"


def sanitize_onnx_names(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace '.' with '_' in ONNX names so torch.export can traverse submodules."""
    graph = model.graph
    rename_map: dict[str, str] = {}
    used_names: set[str] = set()

    def get_clean_name(name: str) -> str:
        if not name:
            return name
        if name in rename_map:
            return rename_map[name]
        clean = name.replace(".", "_")
        while clean in used_names and clean != name:
            clean += "_"
        rename_map[name] = clean
        used_names.add(clean)
        return clean

    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        vi.name = get_clean_name(vi.name)
    for init in graph.initializer:
        init.name = get_clean_name(init.name)
    for node in graph.node:
        node.name = get_clean_name(node.name)
        for i, inp in enumerate(node.input):
            node.input[i] = get_clean_name(inp)
        for i, out in enumerate(node.output):
            node.output[i] = get_clean_name(out)

    return model


ONNX_FLOAT16 = onnx.TensorProto.FLOAT16
ONNX_FLOAT32 = onnx.TensorProto.FLOAT


def promote_fp16_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite FLOAT16 tensors, casts, and value types to FLOAT32 so the graph runs in float32.

    Module-level .to(float32) only converts parameters and buffers. Casts inside the graph
    and tensor constants keep their FP16 type, which breaks float32 operators at trace time.
    """
    graph = model.graph

    def promote_tensor(tensor: onnx.TensorProto) -> None:
        if tensor.data_type == ONNX_FLOAT16:
            array = numpy_helper.to_array(tensor).astype(np.float32)
            promoted = numpy_helper.from_array(array, tensor.name)
            tensor.CopyFrom(promoted)

    def promote_value(value: onnx.ValueInfoProto) -> None:
        if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == ONNX_FLOAT16:
            value.type.tensor_type.elem_type = ONNX_FLOAT32

    for init in graph.initializer:
        promote_tensor(init)
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        promote_value(value)
    for node in graph.node:
        for attr in node.attribute:
            if attr.name == "to" and node.op_type == "Cast" and attr.i == ONNX_FLOAT16:
                attr.i = ONNX_FLOAT32
            elif attr.name == "value" and attr.HasField("t"):
                promote_tensor(attr.t)
    return model


def specialize_static_shape(model: onnx.ModelProto, input_shape: list[int]) -> onnx.ModelProto:
    """Pin the graph input to a static shape and fold constants.

    Shape-driven ops (Shape -> Gather -> Reshape) become constants, which torch.export can trace.
    Without this, dynamic-shape models fail with data-dependent guard errors.
    """
    import onnxslim

    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) != len(input_shape):
        raise ValueError(f"input has rank {len(dims)}, got --input-shape with rank {len(input_shape)}")
    for dim, size in zip(dims, input_shape):
        dim.ClearField("dim_param")
        dim.dim_value = size
    for out in model.graph.output:
        for dim in out.type.tensor_type.shape.dim:
            dim.ClearField("dim_param")
            dim.ClearField("dim_value")
    return onnxslim.slim(model)


def load_onnx_as_torch(
    onnx_path: str | Path,
    dtype: torch.dtype = torch.float32,
    input_shape: list[int] | None = None,
) -> tuple[torch.nn.Module, onnx.ModelProto]:
    onnx_model = onnx.load(str(onnx_path))

    # onnx2pytorch doesn't support training_mode=0 on BatchNormalization
    for node in onnx_model.graph.node:
        if node.op_type == "BatchNormalization":
            kept = [a for a in node.attribute if not (a.name == "training_mode" and a.i == 0)]
            del node.attribute[:]
            node.attribute.extend(kept)

    if dtype == torch.float32:
        onnx_model = promote_fp16_to_fp32(onnx_model)
    if input_shape is not None:
        onnx_model = specialize_static_shape(onnx_model, input_shape)
    onnx_model = sanitize_onnx_names(onnx_model)
    torch_model = convert_checked(onnx_model).to(dtype).eval()
    return torch_model, onnx_model


def compile_onnx_to_pt2(
    onnx_path: str | Path,
    output_pt2_path: str | Path,
    input_shape: list[int] | None = None,
    dtype: torch.dtype = torch.float32,
) -> Path:
    if input_shape is None:
        inp = onnx.load(str(onnx_path), load_external_data=False).graph.input[0]
        input_shape = [
            d.dim_value if d.dim_value > 0 else 1
            for d in inp.type.tensor_type.shape.dim
        ]
    torch_model, onnx_model = load_onnx_as_torch(onnx_path, dtype, input_shape=list(input_shape))

    example_inputs = (torch.randn(*input_shape, dtype=dtype),)

    with torch.no_grad():
        exported = torch.export.export(torch_model, example_inputs)
        torch._inductor.aoti_compile_and_package(
            exported,
            package_path=str(output_pt2_path),
        )
        output = torch_model(*example_inputs)

    if not isinstance(output, torch.Tensor):
        raise TypeError("only single-output models are supported")
    write_model_meta(
        output_pt2_path,
        input_shape=list(input_shape),
        output_shape=list(output.shape),
    )
    return Path(output_pt2_path)


def write_model_meta(pt2_path: str | Path, input_shape: list[int], output_shape: list[int]) -> None:
    """Append shape metadata to the .pt2 zip so runtimes can read it without the ONNX file."""
    meta = {"input_shape": input_shape, "output_shape": output_shape, "dtype": "float32"}
    with zipfile.ZipFile(pt2_path, "a") as zf:
        zf.writestr(META_NAME, json.dumps(meta))


class AOTIModule(torch.nn.Module):
    """nn.Module wrapper around a compiled .pt2 package."""

    def __init__(self, package_path: str | Path):
        super().__init__()
        self.runner = torch._inductor.aoti_load_package(str(package_path))

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.runner(*args)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_shape(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(dim.strip()) for dim in raw.split(",") if dim.strip()]


if __name__ == "__main__":
    import argparse

    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compile an ONNX model into an AOTInductor .pt2 package.")
    parser.add_argument("--input", type=Path, help="Path to input .onnx model")
    parser.add_argument("--output", type=Path, help="Path to output .pt2 package")
    parser.add_argument("--input-shape", type=str, default=None,
                        help="Comma-separated static input shape, e.g. '1,3,224,224'")
    parser.add_argument("--dtype", type=str, default="float32", choices=list(_DTYPE_MAP),
                        help="Compilation dtype (default: float32)")
    args = parser.parse_args()

    if args.input and args.output:
        out = compile_onnx_to_pt2(
            onnx_path=args.input,
            output_pt2_path=args.output,
            input_shape=_parse_shape(args.input_shape),
            dtype=_DTYPE_MAP[args.dtype],
        )
    elif args.input or args.output:
        parser.error("Both --input and --output must be provided together.")
    else:
        out = compile_onnx_to_pt2(
            base_dir / "model/model_fp16_clean.onnx",
            base_dir / "model/model_compiled.pt2",
        )
    print(f"Compiled package saved to: {out}")
