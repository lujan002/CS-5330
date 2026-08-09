"""Write a contact sheet of augmented views next to the clean scan.

Tuning the domain gap by staring at numbers does not work. If the augmented
views look nothing like a webcam crop of a card on a desk, retrieval will look
good in eval and fall over live.

    python3 -m card_match.preview_augment --cards 4 --views 7
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from .common import ARTIFACT_DIR, IMAGE_DIR, INPUT_H, INPUT_W, load_meta
from .dataset import augment, rectify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, default=4)
    parser.add_argument("--views", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(ARTIFACT_DIR / "augment_preview.png"))
    args = parser.parse_args()

    rows = load_meta()
    if not rows:
        print("no images on disk — run download_tcgdex.py first")
        return 1

    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(rows), min(args.cards, len(rows)), replace=False)

    strips = []
    for index in picked:
        scan = cv2.imread(str(IMAGE_DIR / rows[index]["file"]), cv2.IMREAD_COLOR)
        if scan is None:
            continue
        tiles = [rectify(scan)]
        tiles += [augment(scan, rng) for _ in range(args.views)]
        # White rule between the clean scan and its augmented views.
        strip = np.full((INPUT_H, 4, 3), 255, dtype=np.uint8)
        strips.append(
            np.hstack([tiles[0], strip] + [t for tile in tiles[1:] for t in (tile, strip)])
        )

    if not strips:
        print("could not decode any scans")
        return 1

    sheet = np.vstack(
        [s for strip in strips for s in (strip, np.full((4, strips[0].shape[1], 3), 255, np.uint8))]
    )
    cv2.imwrite(args.out, sheet)
    print(f"{len(strips)} cards x {args.views} views -> {args.out}")
    print("left column is the clean gallery scan, the rest are simulated queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
