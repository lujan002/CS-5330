"""Export the cardness classifier to ONNX for ar_card.

Same input contract as the embedder (image, N x 3 x 312 x 224). Output is
N x 2 logits: card, not_card.

    python3 -m card_match.export_cardness_onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .common import (
    CARDNESS_CHECKPOINT_PATH,
    CARDNESS_ONNX_PATH,
    INFERENCE_DIR,
    INPUT_H,
    INPUT_W,
)
from .model import CardnessNet

INPUT_NAME = "image"
OUTPUT_NAME = "logits"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CARDNESS_CHECKPOINT_PATH))
    parser.add_argument("--out", default=str(CARDNESS_ONNX_PATH))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CardnessNet()
    # Prefer early-stopped best weights when the trainer saved them.
    state = checkpoint.get("best_model") or checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    best_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch", "?"))
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, INPUT_H, INPUT_W)

    torch.onnx.export(
        model,
        (example,),
        args.out,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported cardness (best epoch {best_epoch}) -> {args.out}")

    import onnxruntime as ort

    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    batch = torch.randn(3, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        expected = model(batch).numpy()
    actual = session.run([OUTPUT_NAME], {INPUT_NAME: batch.numpy()})[0]

    if actual.shape != (3, CardnessNet.NUM_CLASSES):
        print(
            f"FAIL: expected (3, {CardnessNet.NUM_CLASSES}) output, got {actual.shape}"
        )
        return 1

    drift = float(np.abs(expected - actual).max())
    print(f"torch vs onnxruntime max abs diff {drift:.2e}")
    if drift > 1e-4:
        print("FAIL: onnx graph disagrees with torch")
        return 1
    print("parity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
