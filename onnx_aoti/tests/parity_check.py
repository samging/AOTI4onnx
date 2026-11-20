"""Compare a compiled .pt2 against ONNX Runtime on the same input.

Usage:
    env/bin/python onnx_aoti/tests/parity_check.py \
        --onnx model/model_fp16_clean.onnx \
        --pt2 model/model_compiled.pt2 \
        --shape 1,3,512,768 \
        --atol 1e-2

Exit code is 0 when max |pt2 - onnxruntime| <= atol and the output shapes match.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from onnx_aoti import META_NAME, AOTIModule  # noqa: E402


def read_meta(pt2_path: Path) -> dict | None:
    with zipfile.ZipFile(pt2_path) as zf:
        if META_NAME not in zf.namelist():
            return None
        return json.loads(zf.read(META_NAME))


def parse_shape(raw: str) -> list[int]:
    return [int(dim) for dim in raw.split(",") if dim.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX Runtime parity check for a compiled .pt2 package.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--pt2", type=Path, required=True)
    parser.add_argument("--shape", type=str, default=None,
                        help="Input shape, e.g. 1,3,512,768. Defaults to the shape in the .pt2 metadata.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance (default 1e-3)")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance (default 1e-3)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = read_meta(args.pt2)
    if args.shape:
        shape = parse_shape(args.shape)
    elif meta:
        shape = meta["input_shape"]
    else:
        parser.error("--shape is required when the .pt2 has no model_meta.json")
    if meta and list(meta["input_shape"]) != shape:
        print(f"warning: --shape {shape} differs from compiled input shape {meta['input_shape']}")

    rng = np.random.default_rng(args.seed)
    x = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ref = session.run(None, {input_name: x})[0]

    model = AOTIModule(args.pt2)
    with torch.no_grad():
        out = model(torch.from_numpy(x)).numpy()

    print(f"input shape:  {shape}")
    print(f"ort shape:    {list(ref.shape)}")
    print(f"pt2 shape:    {list(out.shape)}")
    if list(ref.shape) != list(out.shape):
        print("FAIL: output shapes differ")
        return 1

    ref64 = ref.astype(np.float64)
    diff = np.abs(out.astype(np.float64) - ref64)
    scale = float(np.abs(ref).max()) or 1.0
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(f"max_abs_diff:  {max_abs:.3e}")
    print(f"mean_abs_diff: {mean_abs:.3e}")
    print(f"max_rel_diff:  {max_abs / scale:.3e} (relative to max |ort output| = {scale:.3e})")
    print(f"nan/inf in pt2 output: {bool(np.isnan(out).any() or np.isinf(out).any())}")

    allowed = args.atol + args.rtol * np.abs(ref64)
    violations = int(np.count_nonzero(diff > allowed))
    print(f"elements outside atol={args.atol:g} + rtol={args.rtol:g}*|ort|: {violations} of {diff.size}")

    if violations == 0 and not (np.isnan(out).any() or np.isinf(out).any()):
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
