# Project Final — AR Pokemon Card

## Build

```bash
cd project_final
mkdir -p build && cd build
cmake ..
make
```

## Run

```bash
./ar_card
# or: ./ar_card /dev/video0
```

Controls:
- `q` — quit
- `p` — load camera intrinsics and toggle pose/axes on the card
- `s` — print the 4 ordered card corners (TL TR BR BL)

## Intrinsics

Calibrate once with project4 (`'c'` after saving views), then either:

- run `./ar_card` from `project_final/build/` (it looks for `../../project4/build/camera_intrinsics.yaml`), or
- copy `camera_intrinsics.yaml` next to the binary.

## Tuning

Edit thresholds in `detectCardCorners()` inside `data/models/main.cpp` (Canny, min area, approx epsilon).

## Later (TODO in main.cpp)

- Feature matching instead of/in addition to 4 contour corners
- Assimp FBX load + skinned idle animation (no OBJ conversion)
- OCR card name → model; Models Resource download
