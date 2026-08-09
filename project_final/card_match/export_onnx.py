"""Export the trained embedder to ONNX for the C++ matcher.

Only the trunk is exported: image in, L2-normalized vector out. The ArcFace head
stays behind in the checkpoint. The batch axis is dynamic so embed_gallery.py can
push 64 scans at a time while ar_card sends one crop per frame.

    python3 -m card_match.export_onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .common import (
    CHECKPOINT_PATH,
    EMBED_DIM,
    INFERENCE_DIR,
    INPUT_H,
    INPUT_W,
    ONNX_PATH,
)
from .model import CardEmbedder

INPUT_NAME = "image"
OUTPUT_NAME = "embedding"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--out", default=str(ONNX_PATH))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = checkpoint.get("backbone", "mobilenet_v3_large")

    embedder = CardEmbedder(backbone=backbone, pretrained=False)
    embedder.load_state_dict(checkpoint["embedder"])
    embedder.eval()

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, INPUT_H, INPUT_W)

    torch.onnx.export(
        embedder,
        (example,),
        args.out,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {backbone} (epoch {checkpoint.get('epoch', '?')}) -> {args.out}")

    # The C++ side runs onnxruntime, so agreement with onnxruntime is what counts.
    import onnxruntime as ort

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    batch = torch.randn(3, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        expected = embedder(batch).numpy()
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: batch.numpy()})[0]

    if actual.shape != (3, EMBED_DIM):
        print(f"FAIL: expected (3, {EMBED_DIM}) output, got {actual.shape}")
        return 1

    drift = float(np.abs(expected - actual).max())
    norms = np.linalg.norm(actual, axis=1)
    print(f"torch vs onnxruntime max abs diff {drift:.2e}")
    print(f"output norms {norms.round(5).tolist()} (expect 1.0)")

    if drift > 1e-4:
        print("FAIL: onnx graph disagrees with torch")
        return 1
    print("parity ok — next: python3 -m card_match.embed_gallery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
