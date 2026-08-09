"""Shared constants and paths for the Pokemon card matcher.

Anything in here that describes the network input or the gallery file layout is a
contract with the C++ side (src/card_matcher.cpp). Changing a value means
regenerating the artifacts and updating the matching constant in the C++ file.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

# Network input. Portrait, close to the 2.5 x 3.5 in card aspect (0.714); the
# official scans are 600x825 (0.727). Both gallery scans and live camera warps
# are resized to exactly this, so the small aspect difference cancels out.
INPUT_W = 224
INPUT_H = 312

EMBED_DIM = 128

# ImageNet statistics, RGB order, applied after scaling pixels to [0, 1].
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)

# gallery.bin layout, little endian:
#   magic[8] "PKCGAL01"
#   int32 count, int32 dim
#   float32[count * dim]   L2-normalized embeddings, row major
#   count records of: int32 dex, then 4 length-prefixed utf-8 strings
#                     (card_id, name, set_id, set_name)
GALLERY_MAGIC = b"PKCGAL01"

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent

TCG_DIR = PROJECT_ROOT / "data" / "tcg" / "en"
IMAGE_DIR = TCG_DIR / "images"
META_PATH = TCG_DIR / "meta.jsonl"

# data/card_match layout:
#   inference/  — ONNX + gallery.bin consumed by ar_card (ship with the repo)
#   artifacts/  — training checkpoints and debug sidecars (gitignored)
CARD_MATCH_DIR = PROJECT_ROOT / "data" / "card_match"
INFERENCE_DIR = CARD_MATCH_DIR / "inference"
ARTIFACT_DIR = CARD_MATCH_DIR / "artifacts"

CHECKPOINT_PATH = ARTIFACT_DIR / "embedder.pt"
ORIENT_CHECKPOINT_PATH = ARTIFACT_DIR / "orient.pt"
CARDNESS_CHECKPOINT_PATH = ARTIFACT_DIR / "cardness.pt"
GALLERY_NPY_PATH = ARTIFACT_DIR / "gallery_embeddings.npy"
GALLERY_IDS_PATH = ARTIFACT_DIR / "gallery_ids.json"

ONNX_PATH = INFERENCE_DIR / "embedder.onnx"
ORIENT_ONNX_PATH = INFERENCE_DIR / "orient.onnx"
CARDNESS_ONNX_PATH = INFERENCE_DIR / "cardness.onnx"
GALLERY_BIN_PATH = INFERENCE_DIR / "gallery.bin"


def load_meta(path: Path = META_PATH) -> list[dict]:
    """Read meta.jsonl, keeping only rows whose image actually landed on disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run download_tcgdex.py first"
        )
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if (IMAGE_DIR / row["file"]).exists():
                rows.append(row)
    rows.sort(key=lambda r: r["id"])
    return rows


def _write_string(handle, text: str) -> None:
    raw = (text or "").encode("utf-8")
    handle.write(struct.pack("<i", len(raw)))
    handle.write(raw)


def write_gallery_bin(path: Path, embeddings, rows: list[dict]) -> None:
    """Write the single self-contained file the C++ matcher loads.

    Metadata is embedded rather than kept in the sidecar JSON so the C++ side
    needs no JSON parser.
    """
    count, dim = embeddings.shape
    if count != len(rows):
        raise ValueError(f"{count} embeddings but {len(rows)} metadata rows")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(GALLERY_MAGIC)
        handle.write(struct.pack("<ii", count, dim))
        handle.write(embeddings.astype("<f4", copy=False).tobytes(order="C"))
        for row in rows:
            dex = row.get("dex") or []
            handle.write(struct.pack("<i", int(dex[0]) if dex else -1))
            _write_string(handle, row.get("id", ""))
            _write_string(handle, row.get("name", ""))
            _write_string(handle, row.get("set_id", ""))
            _write_string(handle, row.get("set_name", ""))
