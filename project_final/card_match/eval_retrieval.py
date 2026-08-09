"""Retrieval accuracy against synthetically degraded queries.

The gallery is built from clean scans, exactly as embed_gallery.py builds the
shipped index. Queries are the same scans pushed through the training
augmentations, which stand in for a rectified webcam crop. Top-1 here is an
optimistic but useful proxy: it measures robustness to the modelled domain gap,
not to whatever a real camera adds on top.

    python3 -m card_match.eval_retrieval
    python3 -m card_match.eval_retrieval --subset 4000 --views 4
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import CHECKPOINT_PATH, load_meta
from .dataset import CardScanDataset
from .model import CardEmbedder

# Kept away from the seeds train_embedder.py uses so eval views are never
# literally the views the model just trained on.
EVAL_SEED_BASE = 900_000


@torch.no_grad()
def embed_rows(
    embedder: torch.nn.Module,
    rows: list[dict],
    device: torch.device,
    train_aug: bool,
    seed: int = 0,
    batch_size: int = 128,
    workers: int = 8,
) -> np.ndarray:
    dataset = CardScanDataset(rows, train=train_aug, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    was_training = embedder.training
    embedder.eval()

    out = np.empty((len(rows), embedder.embed_dim), dtype=np.float32)
    cursor = 0
    use_amp = device.type == "cuda"
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            vectors = embedder(images)
        vectors = vectors.float().cpu().numpy()
        out[cursor : cursor + len(vectors)] = vectors
        cursor += len(vectors)

    if was_training:
        embedder.train()
    return out


@torch.no_grad()
def run_retrieval_eval(
    embedder: torch.nn.Module,
    rows: list[dict],
    device: torch.device,
    views: int = 2,
    batch_size: int = 128,
    workers: int = 8,
) -> dict:
    gallery = embed_rows(
        embedder, rows, device, train_aug=False, batch_size=batch_size, workers=workers
    )
    gallery_t = torch.from_numpy(gallery).to(device)

    truth = torch.arange(len(rows), device=device)
    top1 = 0
    top5 = 0
    total = 0
    margins = []

    for view in range(views):
        queries = embed_rows(
            embedder,
            rows,
            device,
            train_aug=True,
            seed=EVAL_SEED_BASE + view,
            batch_size=batch_size,
            workers=workers,
        )
        queries_t = torch.from_numpy(queries).to(device)

        # Both sides are L2-normalized, so a dot product is cosine similarity.
        for start in range(0, len(rows), 1024):
            chunk = queries_t[start : start + 1024]
            similarity = chunk @ gallery_t.T
            labels = truth[start : start + 1024]

            best5 = similarity.topk(min(5, similarity.shape[1]), dim=1).indices
            hits = best5 == labels.view(-1, 1)
            top1 += int(hits[:, 0].sum())
            top5 += int(hits.any(dim=1).sum())
            total += len(labels)

            correct = similarity.gather(1, labels.view(-1, 1)).squeeze(1)
            # Gap to the best wrong card: how much room a threshold would have.
            similarity.scatter_(1, labels.view(-1, 1), -2.0)
            margins.append((correct - similarity.max(dim=1).values).cpu())

    return {
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
        "margin": float(torch.cat(margins).mean()) if margins else 0.0,
        "gallery": len(rows),
        "queries": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="evaluate against N cards instead of the whole gallery",
    )
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    embedder = CardEmbedder(
        backbone=checkpoint.get("backbone", "mobilenet_v3_large"), pretrained=False
    )
    embedder.load_state_dict(checkpoint["embedder"])
    embedder.to(device).eval()

    rows = load_meta()
    if args.subset and args.subset < len(rows):
        picked = np.random.default_rng(0).choice(len(rows), args.subset, replace=False)
        rows = [rows[i] for i in sorted(picked)]

    print(f"evaluating {len(rows)} cards x {args.views} views on {device}")
    stats = run_retrieval_eval(
        embedder,
        rows,
        device,
        views=args.views,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(
        f"top1 {stats['top1'] * 100:.2f}%  "
        f"top5 {stats['top5'] * 100:.2f}%  "
        f"margin {stats['margin']:.3f}  "
        f"({stats['queries']} queries vs {stats['gallery']} gallery)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
