# card_match — identifying a card against the English TCG gallery

Offline pipeline that turns every English Pokemon TCG card into a searchable
index, plus an ONNX model `ar_card` loads to identify the card under the camera.
Nothing here runs at capture time: the C++ app only consumes `embedder.onnx` and
`gallery.bin`.

## Why retrieval and not classification

A 21,775-way softmax would work today and be useless in October. Sets keep
coming out, and a new class means retraining. Instead the CNN is trained to map
a card image to a 128-D direction, and identification is a cosine nearest
neighbour against precomputed gallery vectors. Adding a new set is then
`embed_gallery.py`, not a training run.

ArcFace supplies the training signal. It is an ordinary softmax over card ids
with an angular margin pushed between the correct class and the rest, so classes
end up separated by angle rather than merely separable. The classifier weights
are discarded afterwards; only the trunk ships.

The backbone is MobileNetV3-Large (ImageNet pretrained), swappable for ResNet18
or ResNet34 with `--backbone`. No attention layers, so the export is plain ONNX
conv and gemm ops.

## The hard part is the domain gap

Every card has exactly **one** training image: the official top-down scan. There
are no photographs of cards at an angle, under a desk lamp, half-covered in
glare. So all intra-class variation is manufactured in `dataset.augment`, and
each augmentation models a specific way the live query differs from the scan:

| Augmentation | What it stands in for |
|---|---|
| jittered source quad, ±5° rotation, inset/outset | the detected quad is never exactly the card outline |
| resolution round-trip | a card fills ~200x280 px of a 640x480 frame, then gets upsampled |
| exposure, contrast, per-channel gain, hue, gamma | room lighting and white balance |
| directional light ramp | one side of the card is lit more than the other |
| elliptical specular blobs | holo and foil sheen, which is what a webcam actually sees |
| gaussian and motion blur | handheld camera |
| gaussian noise, JPEG blocking | webcam sensor and codec |

Check them before trusting a training run:

```bash
python3 -m card_match.preview_augment --cards 4 --views 7
# writes data/card_match/augment_preview.png
```

Left column is the clean gallery scan; the rest are simulated queries. If they
do not look like something your webcam would produce, retrieval will score well
in eval and fall over live.

### Views have to be fresh every epoch

`CardScanDataset` draws each training view from a worker-local RNG that is never
re-seeded. That looks like a detail and is not.

The obvious alternative — seed the augmentation on `(seed, epoch, index)` and
have the training loop call `dataset.set_epoch()` — is broken with
`persistent_workers=True`, because each DataLoader worker holds its *own* copy of
the dataset and never sees the update. Every epoch then replays byte-identical
images, so there is exactly one fixed view per card and the model memorizes them:

| | augmentations replayed | augmentations fresh |
|---|---|---|
| train accuracy, epoch 10 | 100% | 91% |
| retrieval top-1 | 32% | 99.9% |

Perfect training accuracy with an ArcFace loss of 0.000 is the tell. If that
happens in the first few epochs the model is memorizing, not generalizing.

## Preprocessing is a contract

Gallery vectors and live query vectors have to come out of identical
preprocessing or the cosine comparison is meaningless. The contract lives in
`common.py` and is mirrored by constants at the top of `src/card_matcher.cpp`:

- input `224 x 312` BGR, warped from the card quad
- BGR to RGB, scale to `[0, 1]`, subtract ImageNet mean, divide by ImageNet std
- NCHW float32, output L2-normalized to 128 dims

One trap worth knowing: `cv2.warpPerspective` **cannot** area-sample. Passing
`INTER_AREA` silently gives you bilinear, which aliases badly when a 600x825
scan is squeezed to 224x312. `rectify()` therefore uses a plain area resize when
there is no quad, and pre-shrinks with area averaging before warping when there
is. Getting this wrong costs about 0.01 of cosine on a card matched against its
own scan — enough to notice, and a sign the two sides have drifted apart.

Verify parity any time you touch preprocessing on either side. A gallery scan
fed back through the C++ matcher must return itself at `1.0000`:

```bash
./build/match_image data/tcg/en/images/swsh3-136.webp --top 3
```

## Upright orientation (0° / 180°)

Portrait (short edge on top) is still solved in C++ from aspect ratio. The
remaining upright-vs-upside-down guess used to be a Canny upper/lower heuristic;
that is replaced by a MobileNetV3-Small binary classifier trained on the **same**
224×312 crop pipeline as the embedder.

Official scans are upright. Training applies `dataset.augment`, then flips half
the views 180° and labels them upside-down. Live `ar_card` warps the portrait
candidate with `rectifyCard` to matcher size and scores `logit_upright -
logit_upside_down`.

```bash
python3 -m card_match.train_orientation --epochs 10
python3 -m card_match.export_orientation_onnx
# then: ./build/ar_card --orient data/card_match/orient.onnx
```

If `orient.onnx` is missing, `ar_card` falls back to the old Canny score.

## Full-card vs not-a-card (cardness)

A tiny (~100k param) conv net rejects detector false positives like text-box /
art-window panels that still share card aspect. Same 224×312 input as matching.

- **card** — full portrait crop (0° or 180°; upside-down is still a card)
- **not_card** — random / lower-text / art-window subcrops stretched to the canvas

```bash
python3 -m card_match.train_cardness --epochs 8
python3 -m card_match.export_cardness_onnx
# then: ./build/ar_card --match on --cardness data/card_match/cardness.onnx
```

`ar_card` runs cardness on each oriented detect before embedding. Orientation
only picks upright vs 180°; it no longer rejects quads as non-cards. If
`cardness.onnx` is missing, classical aspect / nesting filters alone apply.

## Running the pipeline

```bash
cd project_final

# 1. Full English gallery: ~21.8k cards, ~1.7 GB. Resumable.
python3 -m card_match.download_tcgdex --with-dex --workers 24

# 2. Train. ~40 s/epoch on an RTX 5060 Ti, so ~55 min for 80 epochs.
python3 -m card_match.train_embedder --epochs 80 --batch-size 128 --workers 12

# 3. Export the trunk; checks torch/onnxruntime agreement.
python3 -m card_match.export_onnx

# 4. Embed every scan through onnxruntime, write the index.
python3 -m card_match.embed_gallery

# 5. Retrieval accuracy against synthetic queries.
python3 -m card_match.eval_retrieval --subset 4000 --views 2
```

Then run the app:

```bash
cd build && ./ar_card --match on
```

## Results

MobileNetV3-Large, 128-D, 80 epochs (~53 min on an RTX 5060 Ti), evaluated
against the full gallery with two augmented queries per card:

| | |
|---|---|
| gallery | 21,775 cards |
| queries | 43,550 |
| top-1 | 98.54% |
| top-5 | 100.00% |
| mean margin over best wrong card | 0.418 |

Score separation, which is what the live threshold is set from:

| | cosine |
|---|---|
| correct card, median | 0.945 |
| correct card, 1st percentile | 0.837 |
| correct card, worst seen | 0.704 |
| not a card (noise, flat colour, scrambled art), worst seen | 0.707 |

`ar_card` therefore rejects anything below 0.70.

About 3.2% of the gallery has a twin at >0.99 cosine, and 98% of those twins
share the same card name — reprints whose artwork is identical across two sets.
Those are indistinguishable from the image alone, so they account for most of the
top-1 gap while still yielding the right card name. `Old Amber` in Genetic Apex
versus Mythical Island is an exact 1.0000 tie.

## Files

| Script | Job |
|---|---|
| `download_tcgdex.py` | every English card image + metadata from TCGdex |
| `dataset.py` | rectification, augmentation, embedder + orientation + cardness Datasets |
| `model.py` | `CardEmbedder`, `ArcFaceHead`, `OrientationNet`, `CardnessNet` |
| `train_embedder.py` | training loop, cosine LR schedule, periodic retrieval eval |
| `train_orientation.py` | binary upright / upside-down CE on shared crops |
| `train_cardness.py` | binary full-card / not-a-card CE on shared crops |
| `export_onnx.py` | trunk to ONNX with a dynamic batch axis, plus a parity check |
| `export_orientation_onnx.py` | orientation classifier to ONNX + parity check |
| `export_cardness_onnx.py` | cardness classifier to ONNX + parity check |
| `embed_gallery.py` | gallery vectors through onnxruntime, writes the index |
| `eval_retrieval.py` | top-1 / top-5 against synthetically degraded queries |
| `preview_augment.py` | contact sheet of augmented views |

## Artifacts

Written to `project_final/data/card_match/` (gitignored):

| File | Consumer |
|---|---|
| `embedder.pt` | training checkpoint, holds the ArcFace head for `--resume` |
| `embedder.onnx` | `ar_card` |
| `orient.pt` | orientation training checkpoint |
| `orient.onnx` | `ar_card` upright / 180° classifier |
| `cardness.pt` | cardness training checkpoint |
| `cardness.onnx` | `ar_card` full-card / not-a-card gate before matching |
| `gallery.bin` | `ar_card` — embeddings and card metadata in one file, no JSON parser needed |
| `gallery_embeddings.npy` | eval and debugging |
| `gallery_ids.json` | human-readable row order |

`gallery.bin` is little-endian: magic `PKCGAL01`, `int32 count`, `int32 dim`,
`float32[count * dim]` row-major L2-normalized embeddings, then per row an
`int32` dex number and four length-prefixed UTF-8 strings (card id, name, set
id, set name).

## Data source

[TCGdex](https://tcgdex.dev) (`api.tcgdex.net/v2/en`), free and keyless. The set
endpoint returns each card's id, name and image base URL, so a full pull is ~220
metadata requests rather than one per card; `--with-dex` adds a per-card request
to pick up national dex numbers, which is what lets a match drive the existing
`--pokemon` model loader later.

Images are `high.webp` (600x825, ~65 KB) — same pixels as `high.png` at a fifth
the size. About 50 cards have no image and are dropped from `meta.jsonl`.

## Adding a new set without retraining

```bash
python3 -m card_match.download_tcgdex   # picks up new sets, skips cached images
python3 -m card_match.embed_gallery     # re-embeds, rewrites gallery.bin
```

The embedder never sees the new cards during training and still places them
sensibly, which is the whole reason for the metric-learning setup. Retrain only
if accuracy on new art drifts.
