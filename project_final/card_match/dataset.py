"""Dataset and augmentation for card-scan to webcam-crop transfer.

There is exactly one training image per card: the official top-down scan. All
intra-class variation therefore has to be manufactured, and every augmentation
here models a specific way the live query differs from that scan.

  geometry     the detected quad is never exactly the card outline, so the source
               corners are jittered and the crop is inset/outset a little
  resolution   a card fills maybe 200x280 px of a 640x480 frame and is then
               upsampled to the network input, so detail is genuinely lost
  photometric  room lighting, white balance, and an uneven light gradient
  glare        holo and foil cards throw a broad specular sheen under a webcam
  sensor       motion blur, JPEG blocking, and shot noise
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import IMAGE_DIR, INPUT_H, INPUT_W, PIXEL_MEAN, PIXEL_STD

# cv2 inside DataLoader workers otherwise fights torch for cores.
cv2.setNumThreads(0)

_MEAN = np.array(PIXEL_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(PIXEL_STD, dtype=np.float32).reshape(3, 1, 1)

_DST_QUAD = np.array(
    [[0, 0], [INPUT_W - 1, 0], [INPUT_W - 1, INPUT_H - 1], [0, INPUT_H - 1]],
    dtype=np.float32,
)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 -> normalized RGB float CHW, matching the C++ preprocessing."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy((chw - _MEAN) / _STD)


def rectify(bgr: np.ndarray, quad: np.ndarray | None = None) -> np.ndarray:
    """Warp a quad (default: the whole image) onto the network input rectangle.

    warpPerspective cannot area-sample -- it silently falls back to bilinear --
    so shrinking a 600x825 scan straight through it aliases badly. The no-quad
    case is therefore a plain area resize, which is also bit-for-bit what
    CardMatcher::match does in C++, and the quad case pre-shrinks with area
    averaging before warping.
    """
    if quad is None:
        return cv2.resize(bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)

    h, w = bgr.shape[:2]
    quad = quad.astype(np.float32)

    # Land near the input size before warping, keeping some headroom so the
    # perspective resample still has detail to work with.
    shrink = min(1.0, 1.5 * max(INPUT_W / max(1, w), INPUT_H / max(1, h)))
    if shrink < 0.95:
        bgr = cv2.resize(
            bgr, (max(1, int(w * shrink)), max(1, int(h * shrink))),
            interpolation=cv2.INTER_AREA,
        )
        quad = quad * shrink

    matrix = cv2.getPerspectiveTransform(quad, _DST_QUAD)
    return cv2.warpPerspective(
        bgr, matrix, (INPUT_W, INPUT_H),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _jittered_source_quad(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Card corners as the detector might report them: inset/outset, rotated, noisy."""
    # Detector quad sits slightly inside or outside the printed border.
    inset = rng.uniform(-0.030, 0.045)
    x0, y0 = w * inset, h * inset
    x1, y1 = w * (1.0 - inset), h * (1.0 - inset)
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    angle = rng.uniform(-5.0, 5.0)
    if abs(angle) > 0.1:
        radians = np.deg2rad(angle)
        cos_a, sin_a = np.cos(radians), np.sin(radians)
        center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        quad = (quad - center) @ rotation.T + center

    # Residual per-corner error left over after smoothing, a few px at 640x480.
    quad += rng.normal(0.0, 0.012, size=quad.shape).astype(np.float32) * np.array(
        [w, h], dtype=np.float32
    )
    return quad


def _degrade_resolution(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Round-trip through the pixel count the card actually occupies on screen."""
    scale = rng.uniform(0.32, 1.0)
    if scale > 0.95:
        return bgr
    small = cv2.resize(
        bgr, (max(24, int(INPUT_W * scale)), max(32, int(INPUT_H * scale))),
        interpolation=cv2.INTER_AREA,
    )
    up = cv2.INTER_LINEAR if rng.random() < 0.7 else cv2.INTER_NEAREST
    return cv2.resize(small, (INPUT_W, INPUT_H), interpolation=up)


def _photometric(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = bgr.astype(np.float32)

    out *= rng.uniform(0.60, 1.35)                                  # exposure
    mean = out.mean()
    out = (out - mean) * rng.uniform(0.70, 1.30) + mean             # contrast

    # Per-channel gain stands in for white balance drift.
    out *= rng.uniform(0.90, 1.10, size=3).astype(np.float32)
    out = np.clip(out, 0, 255)

    if rng.random() < 0.7:
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.65, 1.30), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if rng.random() < 0.5:
        gamma = rng.uniform(0.7, 1.4)
        out = np.power(out / 255.0, gamma) * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


def _light_gradient(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A desk lamp lights one side of the card more than the other."""
    ys, xs = np.mgrid[0:INPUT_H, 0:INPUT_W].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (xs / INPUT_W) * np.cos(angle) + (ys / INPUT_H) * np.sin(angle)
    ramp = (ramp - ramp.min()) / max(1e-6, ramp.max() - ramp.min())
    strength = rng.uniform(0.10, 0.35)
    gain = (1.0 - strength * 0.5 + strength * ramp)[..., None]
    return np.clip(bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _glare(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Broad specular sheen, the way holo foil reads through a webcam."""
    mask = np.zeros((INPUT_H, INPUT_W), dtype=np.float32)
    for _ in range(rng.integers(1, 3)):
        center = (int(rng.uniform(0, INPUT_W)), int(rng.uniform(0, INPUT_H)))
        axes = (
            int(rng.uniform(0.18, 0.75) * INPUT_W),
            int(rng.uniform(0.05, 0.30) * INPUT_H),
        )
        cv2.ellipse(mask, center, axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
    blur = int(rng.uniform(0.10, 0.25) * INPUT_W) | 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0) * rng.uniform(0.15, 0.60)

    out = bgr.astype(np.float32)
    tint = rng.uniform(215, 255, size=3).astype(np.float32)
    out += mask[..., None] * (tint - out)
    return np.clip(out, 0, 255).astype(np.uint8)


def _blur(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        k = int(rng.choice([3, 5, 7]))
        return cv2.GaussianBlur(bgr, (k, k), rng.uniform(0.5, 2.0))
    # Directional smear from hand or camera motion.
    length = int(rng.integers(3, 10))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D(
        (length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1.0
    )
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    total = kernel.sum()
    if total < 1e-6:
        return bgr
    return cv2.filter2D(bgr, -1, kernel / total)


def _sensor_noise(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = bgr
    if rng.random() < 0.6:
        noise = rng.normal(0.0, rng.uniform(2.0, 10.0), size=out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        quality = int(rng.integers(30, 92))
        ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return out


def augment(scan_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Full scan -> a plausible rectified webcam crop of the same card."""
    h, w = scan_bgr.shape[:2]
    out = rectify(scan_bgr, _jittered_source_quad(w, h, rng))
    out = _degrade_resolution(out, rng)
    out = _photometric(out, rng)
    if rng.random() < 0.6:
        out = _light_gradient(out, rng)
    if rng.random() < 0.45:
        out = _glare(out, rng)
    if rng.random() < 0.6:
        out = _blur(out, rng)
    return _sensor_noise(out, rng)


class CardScanDataset(Dataset):
    """One official scan per card; the label is the card's row index.

    seed=None draws a fresh view every call, which is what training wants. Do
    not swap that for a per-epoch seed set from the main process: DataLoader
    workers hold their own copy of this object, so with persistent_workers the
    update never arrives and every epoch replays byte-identical augmentations.
    The model then memorizes one fixed image per card, reports perfect training
    accuracy, and retrieves nothing.

    An int seed makes views a deterministic function of the row index, so eval
    queries are reproducible across runs.
    """

    def __init__(self, rows: list[dict], train: bool = True, seed: int | None = None):
        self.rows = rows
        self.train = train
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _fresh_rng(self) -> np.random.Generator:
        # torch gives each worker its own initial seed, and the stream is never
        # reset afterwards, so views keep varying however the loader is set up.
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _read(self, index: int) -> np.ndarray:
        path = IMAGE_DIR / self.rows[index]["file"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode {path}")
        return image

    def __getitem__(self, index: int):
        image = self._read(index)
        if not self.train:
            view = rectify(image)
        elif self.seed is None:
            view = augment(image, self._fresh_rng())
        else:
            rng = np.random.default_rng((self.seed, index).__hash__() & 0xFFFFFFFF)
            view = augment(image, rng)
        return to_tensor(view), index


class OrientationDataset(Dataset):
    """Same scans and augment as matching; label is upright (0) or upside-down (1).

    Official scans are always upright. After the shared webcam-domain augment, half
    the views are rotated 180 so the classifier sees both orientations under the
    same photometric / geometric noise the matcher already trains on.
    """

    def __init__(self, rows: list[dict], train: bool = True, seed: int | None = None):
        self.rows = rows
        self.train = train
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _fresh_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _read(self, index: int) -> np.ndarray:
        path = IMAGE_DIR / self.rows[index]["file"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode {path}")
        return image

    def __getitem__(self, index: int):
        image = self._read(index)
        if not self.train:
            view = rectify(image)
            # Deterministic half upside-down for a fixed eval split feel.
            flip = (index % 2) == 1
        elif self.seed is None:
            rng = self._fresh_rng()
            view = augment(image, rng)
            flip = bool(rng.random() < 0.5)
        else:
            rng = np.random.default_rng((self.seed, index).__hash__() & 0xFFFFFFFF)
            view = augment(image, rng)
            flip = bool(rng.random() < 0.5)

        if flip:
            view = cv2.rotate(view, cv2.ROTATE_180)
        label = 1 if flip else 0
        return to_tensor(view), label


def _random_partial_crop(scan_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Crop a proper subset of the card and stretch it to the network canvas.

    Mimics a detector locking onto a text box / art window / random panel and
    warping that quad to 224x312 — still card pixels, but not a full card.
    """
    h, w = scan_bgr.shape[:2]

    # Mix free random windows with half-card slabs (the live failure mode).
    mode = rng.random()
    if mode < 0.25:
        # Lower text-ish panel.
        y0, y1 = int(h * rng.uniform(0.45, 0.62)), h
        x0, x1 = int(w * rng.uniform(0.0, 0.08)), int(w * rng.uniform(0.92, 1.0))
    elif mode < 0.40:
        # Upper art window.
        y0, y1 = int(h * rng.uniform(0.08, 0.18)), int(h * rng.uniform(0.48, 0.62))
        x0, x1 = int(w * rng.uniform(0.05, 0.15)), int(w * rng.uniform(0.85, 0.95))
    elif mode < 0.55:
        # Left or right half.
        if rng.random() < 0.5:
            x0, x1 = 0, max(2, int(w * rng.uniform(0.40, 0.55)))
        else:
            x0, x1 = min(w - 2, int(w * rng.uniform(0.45, 0.60))), w
        y0, y1 = int(h * rng.uniform(0.0, 0.10)), int(h * rng.uniform(0.90, 1.0))
    else:
        # Random window: keep area clearly below a full card.
        for _ in range(8):
            fh = rng.uniform(0.22, 0.70)
            fw = rng.uniform(0.28, 0.85)
            if fh * fw <= 0.55:
                break
        else:
            fh, fw = 0.45, 0.70
        ch = max(8, int(h * fh))
        cw = max(8, int(w * fw))
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        y1, x1 = y0 + ch, x0 + cw

    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, max(y0 + 2, y1)), min(w, max(x0 + 2, x1))
    crop = scan_bgr[y0:y1, x0:x1]
    return cv2.resize(crop, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


def _webcam_noise(view: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric / sensor noise on an already-sized 224x312 view."""
    out = _degrade_resolution(view, rng)
    out = _photometric(out, rng)
    if rng.random() < 0.5:
        out = _light_gradient(out, rng)
    if rng.random() < 0.35:
        out = _glare(out, rng)
    if rng.random() < 0.5:
        out = _blur(out, rng)
    return _sensor_noise(out, rng)


class CardnessDataset(Dataset):
    """Full portrait card (label 0) vs random partial card crop (label 1).

    Positives use the same webcam-domain augment as the matcher. Negatives are
    stretched sub-regions (text box, art window, random panels) with the same
    photometric noise so the net learns layout, not compression artifacts.
    Upside-down full cards still count as cards.
    """

    LABEL_CARD = 0
    LABEL_NOT_CARD = 1

    def __init__(self, rows: list[dict], train: bool = True, seed: int | None = None):
        self.rows = rows
        self.train = train
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _fresh_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _read(self, index: int) -> np.ndarray:
        path = IMAGE_DIR / self.rows[index]["file"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"could not decode {path}")
        return image

    def __getitem__(self, index: int):
        image = self._read(index)
        if not self.train:
            is_card = (index % 2) == 0
            rng = np.random.default_rng((0xC4D, index).__hash__() & 0xFFFFFFFF)
        elif self.seed is None:
            rng = self._fresh_rng()
            is_card = bool(rng.random() < 0.5)
        else:
            rng = np.random.default_rng((self.seed, index).__hash__() & 0xFFFFFFFF)
            is_card = bool(rng.random() < 0.5)

        if is_card:
            if self.train:
                view = augment(image, rng)
            else:
                view = rectify(image)
                view = _webcam_noise(view, rng) if self.seed is not None else view
            # Full card upside-down is still a card.
            if rng.random() < 0.5:
                view = cv2.rotate(view, cv2.ROTATE_180)
            label = self.LABEL_CARD
        else:
            view = _random_partial_crop(image, rng)
            if self.train or self.seed is not None:
                view = _webcam_noise(view, rng)
            if rng.random() < 0.5:
                view = cv2.rotate(view, cv2.ROTATE_180)
            label = self.LABEL_NOT_CARD

        return to_tensor(view), label
