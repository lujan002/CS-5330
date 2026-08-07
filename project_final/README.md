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
./ar_card --model pyramid          # wireframe pyramid (default)
./ar_card --model pokemon          # OpenGL Pokemon on the card
./ar_card --model pokemon /dev/video0
./ar_card --model pokemon --scale 0.75             # everything at 75% of battle mesh size
./ar_card --model pokemon --pokemon Gengar         # download Gengar for this run
./ar_card --model pokemon --pokemon Onix --pokemon Lapras
```

Arguments:
- `--model pyramid|pokemon` — which overlay to draw
- `--scale IN_PER_UNIT` — inches on the card per XY mesh unit, default `1.0` (see Scaling)
- `--pokemon NAME` — fetch `NAME` from The Models Resource and add it to the cycle
  list for this run only; repeatable

Controls:
- `q` — quit
- `p` — load camera intrinsics and toggle pose/axes on the card
- `m` — next model. With `--model pokemon` this cycles every model found under
  `data/assets/` plus anything downloaded via `--pokemon`, then back to off;
  with `--model pyramid` it just toggles.
- `s` — print the 4 ordered card corners (TL TR BR BL)

## Models

- `pyramid` — project4-style colored wireframe pyramid via `projectPoints`
- `pokemon` — every model under `data/assets/` is discovered at startup and drawn with
  the OpenGL overlay renderer. The current model's name is drawn in the corner.

When several exports of the same model sit side by side, the preference order is
`*_ColladaMax.DAE` → `*_OpenCollada.DAE` → plain `.dae` → `.obj`. The XY `.FBX` files are
an old format Assimp rejects. OpenCollada is normally a duplicate of the ColladaMax
export, but some rips (Gyarados) ship only that one, so it is a fallback rather than a
blanket skip. Mega Evolutions are filtered out, since they are alternate forms of a
Pokemon that is already in the list.

Gendered rips ship as a pair (`VenusaurM`/`VenusaurF`, `zubat_m`/`zubat_f`). Only the male
model is kept, and it is listed under the plain species name. A lone suffixed model is left
alone: it is the only export available, and a trailing `m`/`f` may just be part of the name.

### Sub-meshes

Sub-meshes are transformed by their node's matrix by hand rather than with Assimp's
`aiProcess_PreTransformVertices`, because that flag merges everything sharing a material and
the names are needed to spot alternate-expression parts.

Some models ship lettered variants of one part for the animation system to choose between.
Gengar has a `TongueA` tucked into its mouth and a `TongueB` lolling out; with no animation
to pick one, drawing both left a tongue hanging through its chin. A shared base name alone
does not condemn a part — Blastoise's `CannonsA`/`CannonsB` are both real — and neither does
sticking out, since Greninja's single `Tongue` scarf juts further (35% of body depth) than
Gengar's tongue does (42%). Only a part that is *both* a lettered variant *and* outside the
body's silhouette by more than a quarter of its extent is dropped.

Because sub-meshes arrive interleaved by material, triangles are bucketed per material
before being written to the index buffer; a draw range is one contiguous slice of it.

### Orientation

The rips disagree about which way is up: the XY exports come out `+Y` up, while the
newer per-Pokemon exports are `-Y` up (Snorlax) or `-Z` up (Haunter, Onix, Lapras).
`detectOrientation()` infers it from the fact that these models are authored standing on
a ground plane — the base of the up axis sits at (or very near) zero, while the other
axes straddle the origin. X is always the bilateral symmetry axis.

Effect meshes are excluded from the bounds this reads. Weezing's gas volume is twice the
size of its body and straddles the origin, which made `+Y` look better grounded than the
body's true `-Z` up axis and left it floating on its side.

The mesh is then rotated so up points along card `-Z`, matching the pyramid overlay. The
rotation is built as `row1 = row2 × row0` so its determinant is `+1` and models are never
mirrored.

### Scaling

The XY Collada rips declare `<unit name="inch"/>` and the meshes are already sized the
way they appear in battle — not Pokédex metres. Caterpie's mesh is ~0.76 units tall and
Charizard's ~4.4; Onix from a newer rip folds down to ~5 units after the inch↔metre
correction, so it only slightly tops Charizard on screen the way it does in X/Y, instead
of the 8.8 m Pokédex figure.

`--scale` is inches on the card per mesh unit (default `1.0`), i.e. use the authored
battle size as-is. At that default: Caterpie/Pidgey ~0.76 in, Pikachu ~1.3 in,
Beedrill ~3.3 in, Charizard/Venusaur ~4.4 in, Snorlax ~5.1 in. Pass `--scale 0.5` to
shrink everything uniformly. Large Pokemon overhang the 2.5 × 3.5 in card.

### Downloading models

`--pokemon NAME` pulls the model from The Models Resource at runtime:

1. the game index (`/3ds/pokemonxy/`) is fetched once and scanned for
   `/3ds/pokemonxy/asset/<id>/` tiles and their `title="#0003 Venusaur"` labels
2. the asset page's download button gives the archive path
3. the zip is downloaded and unpacked into `/dev/shm` (tmpfs, i.e. RAM)
4. the extracted folder goes through the normal discovery rules, so an archive that
   contains both genders adds both

Nothing is written to persistent storage, and the scratch directories are deleted when
the process exits. Names are matched case- and punctuation-insensitively (`"mr mime"`
finds `Mr. Mime`) and a miss suggests near matches.

### Material conventions

Textures are resolved from each diffuse `*1.png` by sibling name, and only used if the
file actually exists:

| Sibling | Use |
| --- | --- |
| `*Nor` | normal map |
| `*Nor_Alpha` | specular |
| `*2` | shade colour |
| `*Mask` | alpha cutout, only when alpha actually varies (not a flat film) |
| `*1_Merged` | pre-composited eye, used instead of `*1` when present |

`*1Id` masks are skipped — they are authored on a different UV layout than the diffuse atlas.

`*Mask` comes in three unrelated flavours. Charizard's `FireStenAMask` is RGBA with a
varying alpha channel -- a genuine cutout. Venusaur's `BodyBMask`/`BodyCMask` are RGB
toon-shading masks whose red channel is ~48% black; using that as coverage erased half of
Venusaur's BodyB mesh, including the trunk under the flower. Beedrill's wing Mask is RGBA
but alpha is stuck at ~93 everywhere -- a flat film opacity, not a shape. So a `*Mask` is
only bound as a cutout when its alpha channel both exists and varies; otherwise it is
ignored.

Two layering rules keep the overlays in the right order:

- Eyes: where a `*Eye1_Merged.png` exists (Roserade, Blastoise) it already has the eyelid
  drawn over the iris, so the coplanar `*Iris1` decal quad is dropped. Models without a
  merged texture (Charizard, Venusaur) just draw their single eye material.
- Particle sheets: a colourless diffuse map with no normal-map sibling is an unlit
  intensity map rather than albedo (Charizard's `FireCoreA`/`FireStenA`). These are tinted
  and blended additively with depth writes off, and drawn after everything else. The loader
  makes this call rather than the renderer, since it also decides which geometry counts as
  body when measuring bounds.
- `*Mask` cutouts: only used when the alpha channel actually varies (Charizard's flame). A
  flat film opacity like Beedrill's wing Mask (alpha stuck at ~93) is ignored so the wings
  stay opaque.
- Color2 shade maps are loaded for diagnostics but not mixed into the soft AR lighting.
  Softly blending those cool-toned toon shades into warm albedos (Raticate's blue-grey
  Body2 over orange Body1) muddied colours under typical card viewing angles.
- Effect volumes: these particle meshes are sized for the move that triggers them, not
  for an idle pose. Weezing's `FireSten` gas volume is a sphere that swallows the whole
  Pokemon and renders as an orange blob. Any additive material whose bounding box is
  larger than the body's is therefore hidden, which keeps localised effects like
  Charizard's tail flame while dropping full-body auras.

Draw order is opaque → alpha cutout → additive.

## Validating assets

`validate_models` loads every discovered model, reports what actually reached the GPU, and
renders each material on its own to count the pixels it contributes. Geometry that imports
but draws nothing is the failure mode that hid Venusaur's trunk, so it is reported as a
failure rather than left for you to spot in the AR view.

```bash
cd build
./validate_models                      # every model under data/assets
./validate_models Charizard Venusaur   # just these
./validate_models --pokemon Gengar     # download and validate too
```

Per model it checks that the mesh loads and has triangles, that every vertex has a UV, that
every resolved texture decodes, that the model as a whole renders, that each drawn material
contributes at least one pixel, and that the final height on the card is plausible. It
writes a montage per model to `build/validation/<name>.png` — full model, a 45° view, then
one tile per material — and exits non-zero if anything failed.

This is what caught the newer rips being clipped by the old 100-unit far plane, Gyarados
shipping only an OpenCollada export, and the `*Mask` misuse.

## Intrinsics

Calibrate once with project4 (`'c'` after saving views), then either:

- run `./ar_card` from `project_final/build/` (it looks for `../../project4/build/camera_intrinsics.yaml`), or
- copy `camera_intrinsics.yaml` next to the binary.

## Tuning

Edit thresholds in `detectCardCorners()` inside `main.cpp` (Canny, min area, approx epsilon).

## Later (TODO in main.cpp)

- Feature matching instead of/in addition to 4 contour corners
- Skinned FBX idle animation (keep Assimp bones; drop PreTransformVertices)
- OCR card name → feed straight into `--pokemon`'s downloader
