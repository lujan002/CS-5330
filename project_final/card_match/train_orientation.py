"""Train a binary upright / upside-down classifier on card crops.

Reuses the matcher input pipeline (augment + to_tensor at 224x312). Official
scans are upright; half of each epoch's views are rotated 180 after augment.

    python3 -m card_match.train_orientation --epochs 10
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import ARTIFACT_DIR, ORIENT_CHECKPOINT_PATH, load_meta
from .dataset import OrientationDataset
from .model import OrientationNet


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def eval_accuracy(
    model: OrientationNet,
    rows: list[dict],
    device: torch.device,
    workers: int,
    seed: int = 0,
) -> float:
    model.eval()
    dataset = OrientationDataset(rows, train=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=min(4, workers),
        pin_memory=device.type == "cuda",
    )
    correct = 0
    seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images).argmax(1)
        correct += int((pred == labels).sum())
        seen += len(labels)
    model.train()
    return correct / max(1, seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="train on N cards only")
    parser.add_argument("--eval-every", type=int, default=2)
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

    print(f"{len(rows)} cards on {device} (orientation binary CE)")

    dataset = OrientationDataset(rows, train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    model = OrientationNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    if args.resume and ORIENT_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            ORIENT_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"resumed from epoch {start_epoch}")

    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(200, total_steps // 20)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    def save(epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            ORIENT_CHECKPOINT_PATH,
        )

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        epoch_start = time.time()

        for batch, (images, labels) in enumerate(loader):
            lr = lr_at(global_step, total_steps, warmup_steps, args.lr)
            optimizer.param_groups[0]["lr"] = lr

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                "cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
            subset = (
                rows
                if not args.eval_size
                else rows[: min(args.eval_size, len(rows))]
            )
            acc = eval_accuracy(model, subset, device, args.workers)
            print(f"  [eval] {len(subset)} cards  acc {acc * 100:.2f}%", flush=True)

    print(f"\ncheckpoint -> {ORIENT_CHECKPOINT_PATH}")
    print("next: python3 -m card_match.export_orientation_onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
