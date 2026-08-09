"""Embed every gallery scan and write the index ar_card searches.

Inference runs through onnxruntime rather than torch on purpose: the shipped
index is then produced by the exact graph and numerics the C++ side will use, so
a query embedding and a gallery embedding are never a fold-difference apart.

Outputs:
    gallery.bin              embeddings + metadata, the only file C++ reads
    gallery_embeddings.npy   same vectors, for eval and debugging
    gallery_ids.json         human-readable row order

    python3 -m card_match.embed_gallery
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from .common import (
    ARTIFACT_DIR,
    GALLERY_BIN_PATH,
    GALLERY_IDS_PATH,
    GALLERY_NPY_PATH,
    IMAGE_DIR,
    ONNX_PATH,
    load_meta,
    write_gallery_bin,
)
from .dataset import rectify, to_tensor
from .export_onnx import INPUT_NAME, OUTPUT_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default=str(ONNX_PATH))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--cpu", action="store_true", help="skip the CUDA provider even if present"
    )
    args = parser.parse_args()

    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(args.onnx, providers=providers)
    print(f"onnxruntime providers: {session.get_providers()}")

    rows = load_meta()
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no gallery images on disk — run download_tcgdex.py first")
        return 1
    print(f"embedding {len(rows)} cards")

    embeddings = None
    kept: list[dict] = []
    batch: list[np.ndarray] = []
    batch_rows: list[dict] = []
    chunks: list[np.ndarray] = []
    started = time.time()

    def flush() -> None:
        if not batch:
            return
        stacked = np.stack(batch).astype(np.float32)
        chunks.append(session.run([OUTPUT_NAME], {INPUT_NAME: stacked})[0])
        kept.extend(batch_rows)
        batch.clear()
        batch_rows.clear()

    for index, row in enumerate(rows):
        scan = cv2.imread(str(IMAGE_DIR / row["file"]), cv2.IMREAD_COLOR)
        if scan is None:
            print(f"  ! skipping undecodable {row['file']}")
            continue
        batch.append(to_tensor(rectify(scan)).numpy())
        batch_rows.append(row)
        if len(batch) >= args.batch_size:
            flush()
            if (index + 1) % 2000 < args.batch_size:
                rate = (index + 1) / max(1e-6, time.time() - started)
                print(f"  {index + 1}/{len(rows)}  {rate:.0f} cards/s")
    flush()

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    # The graph already normalizes; this guards against a future head change
    # silently breaking the "dot product == cosine" assumption in C++.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(GALLERY_NPY_PATH, embeddings)
    write_gallery_bin(GALLERY_BIN_PATH, embeddings, kept)
    with GALLERY_IDS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "row": i,
                    "id": r["id"],
                    "name": r["name"],
                    "set_id": r.get("set_id", ""),
                    "set_name": r.get("set_name", ""),
                    "dex": (r.get("dex") or [None])[0],
                }
                for i, r in enumerate(kept)
            ],
            handle,
            ensure_ascii=False,
            indent=1,
        )

    # Nearest neighbour among the clean scans themselves: how close the two most
    # similar real cards sit. A live match needs to beat this to be trustworthy.
    sample = embeddings[: min(4000, len(embeddings))]
    similarity = sample @ sample.T
    np.fill_diagonal(similarity, -2.0)
    closest = similarity.max(axis=1)

    print(f"\n{len(kept)} x {embeddings.shape[1]} embeddings in {time.time() - started:.0f}s")
    print(f"  {GALLERY_BIN_PATH}  ({GALLERY_BIN_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  {GALLERY_NPY_PATH}")
    print(f"  {GALLERY_IDS_PATH}")
    print(
        f"nearest distinct card cosine: mean {closest.mean():.3f}  "
        f"p99 {np.percentile(closest, 99):.3f}  max {closest.max():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
