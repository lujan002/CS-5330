"""Train the card embedder with ArcFace over card ids.

Every card contributes exactly one scan, so an "epoch" is one augmented view of
each card and all intra-class variation comes from dataset.augment. The head is
thrown away afterwards; only the embedding trunk ships.

    python3 -m card_match.train_embedder --epochs 40
    python3 -m card_match.train_embedder --backbone resnet18 --batch-size 96
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import ARTIFACT_DIR, CHECKPOINT_PATH, load_meta
from .dataset import CardScanDataset
from .eval_retrieval import run_retrieval_eval
from .model import BACKBONES, build


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mobilenet_v3_large", choices=BACKBONES)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument(
        "--head-lr-mult",
        type=float,
        default=10.0,
        help="the ArcFace weights start from noise, the trunk does not",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--arc-scale", type=float, default=32.0)
    parser.add_argument("--arc-margin", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="train on N cards only")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_meta()
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < 2:
        print("not enough cards on disk — run download_tcgdex.py first")
        return 1

    print(f"{len(rows)} cards / classes on {device}")

    dataset = CardScanDataset(rows, train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,          # BatchNorm1d in the head dislikes a stray 1-sample batch
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    embedder, head = build(
        args.backbone, len(rows), args.arc_scale, args.arc_margin, pretrained=True
    )
    embedder.to(device)
    head.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": embedder.parameters(), "lr": args.lr},
            {"params": head.parameters(), "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    if args.resume and CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        embedder.load_state_dict(checkpoint["embedder"])
        if checkpoint.get("num_classes") == len(rows):
            head.load_state_dict(checkpoint["head"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0)
        print(f"resumed from epoch {start_epoch}")

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(500, total_steps // 20)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    def save(epoch: int) -> None:
        torch.save(
            {
                "embedder": embedder.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "backbone": args.backbone,
                "num_classes": len(rows),
                "arc_scale": args.arc_scale,
                "arc_margin": args.arc_margin,
            },
            CHECKPOINT_PATH,
        )

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        embedder.train()
        head.train()

        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        epoch_start = time.time()

        for batch, (images, labels) in enumerate(loader):
            lr = lr_at(global_step, total_steps, warmup_steps, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * args.head_lr_mult

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                embeddings = embedder(images)
                logits = head(embeddings, labels)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach()) * len(labels)
            running_correct += int((logits.detach().argmax(1) == labels).sum())
            running_seen += len(labels)
            global_step += 1

            if batch % 50 == 0:
                print(
                    f"  epoch {epoch + 1}/{args.epochs} "
                    f"step {batch}/{steps_per_epoch} "
                    f"loss {running_loss / max(1, running_seen):.3f} "
                    f"train-acc {running_correct / max(1, running_seen) * 100:.1f}% "
                    f"lr {lr:.2e}",
                    flush=True,
                )

        print(
            f"epoch {epoch + 1}/{args.epochs} done in {time.time() - epoch_start:.0f}s  "
            f"loss {running_loss / max(1, running_seen):.3f}  "
            f"train-acc {running_correct / max(1, running_seen) * 100:.1f}%",
            flush=True,
        )
        save(epoch + 1)

        last = epoch + 1 == args.epochs
        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or last):
            subset = rows if not args.eval_size else rows[: min(args.eval_size, len(rows))]
            stats = run_retrieval_eval(
                embedder, subset, device, views=1, workers=args.workers
            )
            print(
                f"  [eval] {stats['gallery']} cards  "
                f"top1 {stats['top1'] * 100:.2f}%  "
                f"top5 {stats['top5'] * 100:.2f}%  "
                f"margin {stats['margin']:.3f}",
                flush=True,
            )

    print(f"\ncheckpoint -> {CHECKPOINT_PATH}")
    print("next: python3 -m card_match.export_onnx && python3 -m card_match.embed_gallery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
