"""Train a tiny full-card vs not-a-card classifier.

Positives: webcam-augmented full portrait cards (0° / 180° still count as cards).
Negatives: random / text-box / art-window crops stretched to 224x312.

Stops when validation loss rises for --patience epochs (keeps the best checkpoint).

    python3 -m card_match.train_cardness --epochs 40 --patience 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

# Headless: cv2's Qt plugins otherwise abort when matplotlib touches a display.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common import ARTIFACT_DIR, CARDNESS_CHECKPOINT_PATH, load_meta
from .dataset import CardnessDataset
from .model import CardnessNet

HISTORY_PATH = ARTIFACT_DIR / "cardness_history.json"
PLOT_PATH = ARTIFACT_DIR / "cardness_train.png"


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def split_rows(
    rows: list[dict], val_size: int, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Hold out distinct cards for validation (not just different augments)."""
    if val_size <= 0 or val_size >= len(rows):
        raise ValueError(f"val_size must be in 1..{len(rows) - 1}")
    order = list(range(len(rows)))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(rows), generator=rng).tolist()
    val_idx = set(perm[:val_size])
    train_rows = [rows[i] for i in order if i not in val_idx]
    val_rows = [rows[i] for i in perm[:val_size]]
    return train_rows, val_rows


@torch.no_grad()
def evaluate(
    model: CardnessNet,
    rows: list[dict],
    device: torch.device,
    criterion: nn.Module,
    workers: int,
    seed: int = 0,
) -> dict[str, float]:
    model.eval()
    dataset = CardnessDataset(rows, train=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=min(4, workers),
        pin_memory=device.type == "cuda",
    )
    total_loss = 0.0
    correct = 0
    seen = 0
    card_ok = card_n = 0
    not_ok = not_n = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss) * len(labels)
        pred = logits.argmax(1)
        correct += int((pred == labels).sum())
        seen += len(labels)
        for p, y in zip(pred.tolist(), labels.tolist()):
            if y == CardnessDataset.LABEL_CARD:
                card_n += 1
                card_ok += int(p == y)
            else:
                not_n += 1
                not_ok += int(p == y)
    model.train()
    return {
        "loss": total_loss / max(1, seen),
        "acc": correct / max(1, seen),
        "card_rec": card_ok / max(1, card_n),
        "not_rec": not_ok / max(1, not_n),
    }


def plot_history(history: list[dict], path: Path, best_epoch: int | None = None) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", color="#1f77b4",
            label="train loss")
    val_rows = [h for h in history if h.get("val_loss") is not None]
    if val_rows:
        ax.plot(
            [h["epoch"] for h in val_rows],
            [h["val_loss"] for h in val_rows],
            marker="s",
            color="#d62728",
            label="val loss",
        )
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#2ca02c", linestyle="--", alpha=0.7,
                   label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Cardness loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, [100 * h["train_acc"] for h in history], marker="o",
            color="#1f77b4", label="train acc")
    if val_rows:
        ev = [h["epoch"] for h in val_rows]
        ax.plot(ev, [100 * h["val_acc"] for h in val_rows], marker="s",
                color="#2ca02c", label="val acc")
        ax.plot(ev, [100 * h["val_card_rec"] for h in val_rows], marker="^",
                color="#ff7f0e", label="card recall")
        ax.plot(ev, [100 * h["val_not_rec"] for h in val_rows], marker="v",
                color="#d62728", label="not-card recall")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#2ca02c", linestyle="--", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 105)
    ax.set_title("Cardness accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40,
                        help="max epochs (early stop may finish sooner)")
    parser.add_argument("--patience", type=int, default=5,
                        help="stop after this many evals without val-loss improvement")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="val-loss must drop by at least this to count as improvement")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="train on N cards only")
    parser.add_argument("--val-size", type=int, default=2000,
                        help="held-out cards for validation")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_meta()
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < args.val_size + 2:
        print(f"need at least {args.val_size + 2} cards on disk")
        return 1

    train_rows, val_rows = split_rows(rows, args.val_size, seed=args.split_seed)
    n_params = sum(p.numel() for p in CardnessNet().parameters())
    print(
        f"{len(train_rows)} train / {len(val_rows)} val cards on {device} "
        f"(cardness CE, {n_params:,} params, patience={args.patience})"
    )

    dataset = CardnessDataset(train_rows, train=True)
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

    model = CardnessNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    bad_evals = 0

    if args.resume and CARDNESS_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            CARDNESS_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        if checkpoint.get("best_model") is not None:
            best_state = checkpoint["best_model"]
        if HISTORY_PATH.exists():
            history = json.loads(HISTORY_PATH.read_text())
            history = [h for h in history if h["epoch"] <= start_epoch]
        print(f"resumed from epoch {start_epoch} (best val_loss={best_val_loss:.4f} @ {best_epoch})")

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
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_model": best_state,
            },
            CARDNESS_CHECKPOINT_PATH,
        )

    global_step = start_epoch * steps_per_epoch
    stopped_early = False

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

        train_loss = running_loss / max(1, running_seen)
        train_acc = running_correct / max(1, running_seen)
        print(
            f"epoch {epoch + 1}/{args.epochs} done in {time.time() - epoch_start:.0f}s  "
            f"loss {train_loss:.3f}  train-acc {train_acc * 100:.1f}%",
            flush=True,
        )

        row: dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": None,
            "val_acc": None,
            "val_card_rec": None,
            "val_not_rec": None,
        }

        last = epoch + 1 == args.epochs
        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or last):
            metrics = evaluate(
                model, val_rows, device, criterion, args.workers, seed=args.split_seed
            )
            row["val_loss"] = metrics["loss"]
            row["val_acc"] = metrics["acc"]
            row["val_card_rec"] = metrics["card_rec"]
            row["val_not_rec"] = metrics["not_rec"]
            print(
                f"  [val] {len(val_rows)} cards  "
                f"loss {metrics['loss']:.4f}  "
                f"acc {metrics['acc'] * 100:.2f}%  "
                f"card-rec {metrics['card_rec'] * 100:.2f}%  "
                f"not-rec {metrics['not_rec'] * 100:.2f}%",
                flush=True,
            )

            if metrics["loss"] < best_val_loss - args.min_delta:
                best_val_loss = metrics["loss"]
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_evals = 0
                print(
                    f"  [best] val_loss {best_val_loss:.4f} @ epoch {best_epoch} "
                    f"(acc {metrics['acc'] * 100:.2f}%)",
                    flush=True,
                )
            else:
                bad_evals += 1
                print(
                    f"  [early-stop] no val_loss improve ({bad_evals}/{args.patience})",
                    flush=True,
                )
                if bad_evals >= args.patience:
                    stopped_early = True

        history.append(row)
        save(epoch + 1)
        HISTORY_PATH.write_text(json.dumps(history, indent=2))
        plot_history(history, PLOT_PATH, best_epoch=best_epoch or None)

        if stopped_early:
            print(
                f"\nearly stop at epoch {epoch + 1}: "
                f"val_loss rising (best was epoch {best_epoch})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        save(best_epoch)
        print(f"restored best weights from epoch {best_epoch} (val_loss {best_val_loss:.4f})")
        # Final val pass on the restored weights for a clean report line.
        metrics = evaluate(
            model, val_rows, device, criterion, args.workers, seed=args.split_seed
        )
        print(
            f"best model val  loss {metrics['loss']:.4f}  "
            f"acc {metrics['acc'] * 100:.2f}%  "
            f"card-rec {metrics['card_rec'] * 100:.2f}%  "
            f"not-rec {metrics['not_rec'] * 100:.2f}%",
            flush=True,
        )
        plot_history(history, PLOT_PATH, best_epoch=best_epoch)

    print(f"\ncheckpoint -> {CARDNESS_CHECKPOINT_PATH}")
    print(f"history    -> {HISTORY_PATH}")
    print(f"plot       -> {PLOT_PATH}")
    print("next: python3 -m card_match.export_cardness_onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
