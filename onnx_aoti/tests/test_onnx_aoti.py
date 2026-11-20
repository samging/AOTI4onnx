import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from onnx_aoti import AOTIModule, compile_onnx_to_pt2  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.fc(x.flatten(1))


def export_onnx(model: nn.Module, shape, path: Path):
    torch.onnx.export(
        model,
        (torch.randn(*shape),),
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )


def main():
    torch.manual_seed(0)
    shape = (1, 3, 8, 8)

    model = SmallNet().eval()
    onnx_path = OUT_DIR / "small_net.onnx"
    pt2_path = OUT_DIR / "small_net.pt2"
    export_onnx(model, shape, onnx_path)

    compile_onnx_to_pt2(onnx_path, pt2_path, input_shape=list(shape))
    assert pt2_path.exists(), "pt2 package was not created"
    print(f"[ok] compiled package: {pt2_path}")

    compiled = AOTIModule(pt2_path)
    x = torch.randn(*shape)
    with torch.no_grad():
        ref = model(x)
        out = compiled(x)

    assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
    max_diff = (out - ref).abs().max().item()
    print(f"[info] max abs diff vs PyTorch reference: {max_diff:.2e}")
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), "output mismatch"
    print("[ok] AOTI output matches PyTorch reference")


if __name__ == "__main__":
    main()
