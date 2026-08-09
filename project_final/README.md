# Project Final — AR Pokemon Card

## Build

```bash
cd project_final
mkdir -p build && cd build
cmake ..
make
```

Needs OpenCV 4, OpenGL/GLEW/glfw/X11, Assimp, libcurl, libzip, and a prebuilt
[ONNX Runtime](https://github.com/microsoft/onnxruntime/releases). CMake looks
for the release tree next to the course projects
(`../onnxruntime-linux-x64-1.26.0`); point it elsewhere with
`cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-X.Y.Z`.

## Run

```bash
./ar_card --model pokemon          # textured Pokemon on the card (default)
./ar_card --model pyramid          # wireframe pyramid
./ar_card --model pokemon /dev/video0
./ar_card --model pokemon --scale 0.75             # everything at 75% of log-scale size
./ar_card --model pokemon --scale-y 0.6            # flatter size curve (giants closer to small)
./ar_card --pokemon Gengar         # download Gengar for this run (--model pokemon is default)
./ar_card --pokemon Onix --pokemon Lapras
./ar_card --models poke-3D --pokemon Charizard     # Pokemon-3D-api GLB instead of Models Resource
./ar_card --models poke-3D --match on              # dex-driven overlay from poke-3D GLBs
./ar_card --match on                               # identify the card on screen
```

Arguments:
- `--model pyramid|pokemon` — which overlay to draw (default `pokemon`)
- `--models models-resource|poke-3D` — where 3D meshes come from (default `models-resource`).
  `models-resource` uses bundled XY DAEs under `data/assets/` plus The Models Resource
  downloads. `poke-3D` fetches Assimp-prepared GLBs from
  [Pokemon-3D-api/assets](https://github.com/Pokemon-3D-api/assets) (see below)
- `--scale MULT` — global multiplier on Pokédex log-scale height, default `1.0` (see Scaling)
- `--scale-y Y` — exponent in `log1p(height_m^Y)`, default `0.85` (lower = flatter sizes)
- `--pokemon NAME` — fetch `NAME` for this run only (source depends on `--models`);
  repeatable
- `--match on|off` — identify the card against the English TCG gallery, default `off`
  (see Card identification)
- `--embedder PATH` / `--gallery PATH` — override the matcher artifacts

Controls:
- `q` — quit
- `p` — load camera intrinsics and toggle pose/axes on the card
- `m` — next model. With Pokemon overlay this cycles every model found under
  `data/assets/` plus anything downloaded via `--pokemon`, then back to off;
  with `--model pyramid` it just toggles.
- `s` — print the 4 ordered card corners (TL TR BR BL)
- `c` — print the ranked gallery candidates behind the displayed match

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

The rips disagree about which way is up, and poke-3D disagrees with itself: most of its
assets are `+Y` up, but Mudsdale, Grimer and Rotom's appliance forms are `+Z`, Pikachu is
`-Y`, and Charmander is `-Z`. One code path handles every source — there is no per-rip
special casing, because a rule tuned for one family silently breaks the other.

`detectOrientation()` scores the four candidates (`±Y`, `±Z`; X is always the bilateral
symmetry axis) on the fact that these models are authored standing on a ground plane, so
the base of the true up axis sits at or very near zero. Two corrections make that usable:

- **Eyes veto lying-down poses.** A back or belly plane that happens to sit at zero makes
  the wrong axis look planted, which is what laid Mewtwo and Greedent face-down on the
  card. No character stands with its eyes in the bottom sixth of its own silhouette, so
  any candidate that would put them there is discarded before the planted test runs. The
  eye centroid comes from the eye materials; rips that paint the eyes into the body
  texture simply skip this step.
- **Centroid height veto.** Same idea without eye materials: a candidate that puts the
  body centroid in the bottom 30% of the AABB is standing on the heavy end (Articuno's
  long-axis tip plant, which made it dive with the tail streamers as "up").
- **Tip-plant veto.** A face that merely grazes the origin can score as perfectly planted
  while covering almost no footprint cells (Pikachu's back tip on `-Z`). Candidates with
  a planted score under 0.10 and fewer than 10 occupied base cells are discarded.
- **Continuous pitch refinement.** When every axis-aligned candidate is unplanted, the
  mesh may sit on a diagonal (Articuno's body is ~45° between `+Y` and `-Z`). A short
  search around the discrete answer accepts a nearby YZ angle that actually plants;
  Charizard stays on `+Y` because no nearby angle plants either.
- **Unplanted models fall back to the convention.** Charizard is authored floating, so no
  face is near the origin and the winning score is just the least-bad accident. With no
  ground plane and no eyes to appeal to, `+Y` is the only real evidence left — both glTF
  and the XY rips use it.

A few poke-3D assets (Breloom, Toxicroak) ship every mesh node with the same non-axis-
aligned root rotation, so the character is permanently tilted once transforms are baked.
`undoSharedTiltedRoot()` detects that shared tilt, undoes it, then re-plants the mesh on
`+Y` using contact at both ends of the silhouette (so Breloom stands on its feet instead
of its mushroom cap).

Effect meshes are excluded from the bounds this reads. Weezing's gas volume is twice the
size of its body and straddles the origin, which made `+Y` look better grounded than the
body's true `-Z` up axis and left it floating on its side.

This is a heuristic over inconsistent third-party assets, not a guarantee: it agrees with
a hand-labelled set of 42 models on the common cases; the centroid and tip-plant vetoes
cover the Articuno / Pikachu misses that the eye test alone could not.

The mesh is then rotated so up points along card `-Z`, matching the pyramid overlay. The
rotation is built as `row1 = row2 × row0` so its determinant is `+1` and models are never
mirrored.

### Resting on the card

The pivot rests the *body* bounding box on the card, but that box only covers materials
that survived effect classification, and a hunched or reclining pose can put a limb below
it. After placement every drawn vertex is checked against the card plane, and the model is
lifted until its lowest one just touches: nothing is ever left intersecting the ground.

### Scaling

On-card size comes from **Pokédex height**, not authored mesh proportions. Mesh
rips disagree wildly about absolute scale (Sudowoodo / Abomasnow / Wailord can
tower over everything when left as authored), so each model is normalized to a
target height derived from `data/pokemon_heights.csv` (PokeAPI / veekun dump,
height in decimetres → metres):

```text
target_in = gain * log1p( height_m ^ y )   # then clamp to ~0.55–6.0 in
scale     = target_in / mesh_body_height
```

`gain` is chosen so Pikachu (0.4 m) lands at ~1.3 in when `--scale 1`. The
exponent `y` (`--scale-y`, default `0.85`) controls contrast: lower values pull
giants toward mid-size Pokémon. `--scale` is a global multiplier on the target
after the log map.

At the defaults, rough on-card heights: Caterpie/Pikachu ~1.0–1.3 in, Sudowoodo
~2.7 in, Charizard ~3.2 in, Abomasnow/Snorlax ~3.5–4 in, Wailord/Onix capped near
6 in. Pass `--scale 0.5` to shrink everything uniformly. Large Pokémon still
overhang the 2.5 × 3.5 in card.

### Downloading models

`--pokemon NAME` pulls a model at runtime. The source depends on `--models`
(default `models-resource`).

#### Models Resource (default)

1. the game index (`/3ds/pokemonxy/`) is fetched once and scanned for
   `/3ds/pokemonxy/asset/<id>/` tiles and their `title="#0003 Venusaur"` labels
2. the asset page's download button gives the archive path
3. the zip is downloaded and unpacked into `/dev/shm` (tmpfs, i.e. RAM)
4. the extracted folder goes through the normal discovery rules, so an archive that
   contains both genders adds both

Nothing is written to persistent storage, and the scratch dirs are deleted when
the process exits. Names are matched case- and punctuation-insensitively (`"mr mime"`
finds `Mr. Mime`) and a miss suggests near matches.

#### poke-3D (Pokemon-3D-api GLBs)

`--models poke-3D` downloads GLBs from the
[Pokemon-3D-api/assets](https://github.com/Pokemon-3D-api/assets) repo. Those files
ship Draco-compressed with WebP textures; Assimp cannot load them raw, so each
download is run through `tools/poke3d_prep/prep_glb.mjs` (decode Draco, strip
skins, rewrite textures as PNG) before import.

The matched card **name** selects a form when the assets repo has one: Rotom
appliances (`multiform/RotomHeat.glb`, …), regional folders (`alolan/`,
`hisuian/`, `galar/`), Mega/`x`/`y`, origin, primal, and gmax. Otherwise it
falls back to `regular/<dex>.glb` (then gendered `-M`).

One-time setup:

```bash
cd project_final/tools/poke3d_prep
npm install
```

`--pokemon NAME` resolves the national dex via [PokeAPI](https://pokeapi.co/), then
fetches that dex's GLB (with the same form heuristics as match mode). With
`--match on`, the matched card's dex **and name** drive the download (bundled XY
`data/assets` are skipped). Scratch files live in `/dev/shm` and are deleted on
exit, same as Models Resource.

Smoke-test a single species without the camera:

```bash
cd build
./validate_models --models poke-3D --pokemon Charizard
```

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
  merged texture (Charizard, Venusaur) just draw their single eye material. poke-3D GLBs
  keep both the sclera/lid atlas and the iris: the atlas is split into a sclera
  cutout (back) and eyelid cutout (front), with the iris drawn between them
  (poke-3D pupils live on the iris mesh, unlike XY `Eye1_Merged`).
- Particle sheets: a colourless diffuse map with no normal-map sibling is an unlit
  intensity map rather than albedo (Charizard's `FireCoreA`/`FireStenA`, and the
  same greyscale fire sheets inside poke-3D GLBs). These are tinted and blended
  additively with depth writes off, and drawn after everything else. The loader
  makes this call rather than the renderer, since it also decides which geometry
  counts as body when measuring bounds.

  Face parts are exempt by name: poke-3D irises are greyscale often enough to trip
  the test, and an additively-drawn pupil disappears (Psyduck, Porygon2). Anything
  promoted into the eye stack has the flag cleared for the same reason.

  The rule also runs on untextured materials whose name marks them as an effect.
  Rotom's appliance forms wrap the body in a `FireMask` shell with baseColor alpha
  0.33; alpha-blending it darkened the whole Pokemon and made it look see-through,
  where the game drives it with a fire shader. It now adds a glow in its own
  baseColor instead.
- Face decals that never needed splitting into an eye stack (Mewtwo's `l_eye`) are
  flat quads sitting exactly on the skull. Coplanar geometry loses `GL_LESS`, so
  they drew nothing at all; they now render after the head with a camera-ward
  polygon offset.
- Untextured materials with baseColor alpha < 1 (Solosis / Duosion jelly shells)
  keep that opacity and draw in the alpha pass instead of as solid Kd green.
- Diffuse alpha is only treated as a cutout when it spans near-0 to near-255.
  Mid-range film opacity on eye atlases (Greedent) is ignored so the socket
  does not shade as a darker polygon on the face. The same trap hits whole body
  atlases: Greedent/Skwovet sit around alpha 150, Yanmega's wings are stuck at 92.
  Blending is always on, so writing that alpha made solid fur look ghostly. The
  shader now only honours texture alpha when `loadTexture` classified a real
  cutout; otherwise fragment alpha is forced to the material opacity.
- `*Mask` cutouts: only used when the alpha channel actually varies (Charizard's flame). A
  flat film opacity like Beedrill's wing Mask (alpha stuck at ~93) is ignored so the wings
  stay opaque.
- Color2 shade maps are loaded for diagnostics but not mixed into the soft AR lighting.
  Softly blending those cool-toned toon shades into warm albedos (Raticate's blue-grey
  Body2 over orange Body1) muddied colours under typical card viewing angles.
- Effect volumes: in the XY rips these particle meshes are sized for the move that
  triggers them, not for an idle pose. Weezing's `FireSten` gas volume is a sphere that
  swallows the whole Pokemon and renders as an orange blob. Any additive material whose
  bounding box is larger than the body's is therefore hidden, which keeps localised
  effects like Charizard's tail flame while dropping full-body auras.

  poke-3D is exempt from that cull: its effect geometry is part of the silhouette rather
  than a move volume, and Rapidash's mane and tail are one fire sheet taller than the
  body it attaches to — the size test deleted the flame that makes it a Rapidash.

Draw order is opaque → alpha cutout → additive.

## Card identification

`--match on` names the card under the camera by comparing it against all ~21.8k
English cards. The detector already produces an ordered TL/TR/BR/BL quad, so the
card face is warped to an upright 224x312 rectangle — the live equivalent of the
official top-down scan — and that crop is embedded by a small CNN and matched by
cosine similarity against precomputed gallery vectors. The live warp opens in the
`Card` window; the official gallery scan of the voted match opens beside it in
`Match`. The Video overlay shows the card name, national dex number (when the
card is a Pokemon), TCGdex id, and cosine score.

When a Pokemon card is identified, the 3D overlay is driven from that national
dex (and card name, for poke-3D forms). With the default `--models models-resource`,
local `data/assets` folders tagged `#NNNN` are tried first, otherwise the model
is fetched from The Models Resource for this session only. With `--models poke-3D`,
the overlay comes from the Pokemon-3D-api GLB for that dex/form. Bundled
`--model pokemon` cycling and `--pokemon NAME` are skipped in match mode; `m`
only toggles the overlay on and off. Trainer/Energy cards (no dex) clear the mesh.

The homography only holds for the one plane it was fitted to, the card face.
Fingers over a corner, a bent card, or a quad that has slipped onto the sleeve
all warp to nonsense, so the crop is worth watching in the `Card` window when a
match looks wrong.

A single frame gets spoiled often enough by glare or motion blur that the
displayed answer is a majority vote over the last 7 matches (run every 3rd
frame), and top-1 results below 0.70 cosine are discarded rather than voted on.
That cut comes from measurement, not taste: correctly matched cards score 0.70 at
worst and 0.945 median, while crops that are not cards at all top out at 0.707.
Press `c` to see the ranked candidates behind the vote.

Against all 21,775 cards, retrieval is 98.5% top-1 and 100% top-5 on
synthetically degraded queries. Most of the top-1 misses are not really misses:
about 3% of the gallery is reprint artwork that is pixel-identical across two
sets, so the card *id* is genuinely ambiguous from the image while the card name
still comes out right.

The model and the index are built offline and are not in the repo:

```bash
python3 -m card_match.download_tcgdex --with-dex   # ~21.8k images, ~1.7 GB
python3 -m card_match.train_embedder --epochs 80
python3 -m card_match.export_onnx
python3 -m card_match.embed_gallery
```

`ar_card` then picks up `data/card_match/embedder.onnx` and
`data/card_match/gallery.bin` on its own. Without them `--match on` prints why it
could not start and the rest of the app runs unchanged.

See [card_match/README.md](card_match/README.md) for the architecture, the
augmentation that covers the scan-to-webcam gap, and the preprocessing contract
the two sides share.

To check a still image instead of a camera:

```bash
cd build
./match_image ../data/tcg/en/images/swsh3-136.webp --top 5
./match_image photo.jpg --corners 120,80,410,95,400,520,110,505
```

Feeding it a gallery scan is also the parity test between the Python and C++
preprocessing: the card must come back as itself at `1.0000`.

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

Calibrate once from `project_final/build/`:

```bash
./calibrate_camera [/dev/videoN]
```

Same controls as project4: `'s'` to save views (≥5), `'c'` to calibrate and write
`camera_intrinsics.yaml` next to the binary, `'p'` to check pose. `ar_card` also
still looks under `../../project4/build/camera_intrinsics.yaml` if the local file
is missing.

## Tuning

Edit thresholds in `detectCardCorners()` inside `main.cpp` (Canny, min area, approx epsilon).

## Later (TODO in main.cpp)

- Feature matching instead of/in addition to 4 contour corners
- Skinned FBX idle animation (keep Assimp bones; drop PreTransformVertices)
