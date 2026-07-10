#!/usr/bin/env python3
"""Compute and visualize a 3D RGB color histogram from an image."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VoxelBin:
    r0: float
    g0: float
    b0: float
    dr: float
    dg: float
    db: float
    r_center: float
    g_center: float
    b_center: float
    count: float
    bin_color: tuple[float, float, float]
    opacity: float


def resolve_num_bins(bin_size: int | None, num_bins: int | None) -> int:
    if bin_size is not None and num_bins is not None:
        raise ValueError("Use either --bin-size or --bins, not both.")
    if bin_size is not None:
        if bin_size <= 0 or bin_size > 256:
            raise ValueError("--bin-size must be in the range (0, 256].")
        return int(np.ceil(256 / bin_size))
    if num_bins is not None:
        if num_bins <= 0:
            raise ValueError("--bins must be positive.")
        return num_bins
    return 8


def compute_3d_histogram(
    image_bgr: np.ndarray, num_bins: int, normalize: bool
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build a 3D histogram over B, G, R channels."""
    b, g, r = cv2.split(image_bgr)
    pixels = np.stack([b.ravel(), g.ravel(), r.ravel()], axis=1).astype(np.float32)

    bin_edges = [np.linspace(0, 256, num_bins + 1) for _ in range(3)]
    hist, _ = np.histogramdd(pixels, bins=bin_edges)

    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()

    return hist.astype(np.float64), bin_edges


def _frequency_ratio(count: float, mode: float) -> float:
    return float(count / mode) if mode > 0 else 0.0


def _bin_color(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Full RGB at the bin center (hue encodes which colors are in the bin)."""
    return (r / 255.0, g / 255.0, b / 255.0)


def _display_opacity(count: float, mode: float) -> float:
    """Opacity linear in count relative to the mode bin (mode = 100% opaque)."""
    if mode <= 0:
        return 0.0
    return float(np.clip(count / mode, 0.0, 1.0))


def _extract_voxels(
    hist: np.ndarray,
    bin_edges: list[np.ndarray],
    min_fraction: float,
) -> list[VoxelBin] | None:
    mode = float(hist.max())
    if mode <= 0:
        return None

    b_edges, g_edges, r_edges = bin_edges
    threshold = min_fraction * mode
    voxels: list[VoxelBin] = []

    for ib in range(hist.shape[0]):
        for ig in range(hist.shape[1]):
            for ir in range(hist.shape[2]):
                count = float(hist[ib, ig, ir])
                if count <= threshold:
                    continue

                r0, r1 = float(r_edges[ir]), float(r_edges[ir + 1])
                g0, g1 = float(g_edges[ig]), float(g_edges[ig + 1])
                b0, b1 = float(b_edges[ib]), float(b_edges[ib + 1])
                r_center = (r0 + r1) * 0.5
                g_center = (g0 + g1) * 0.5
                b_center = (b0 + b1) * 0.5

                voxels.append(
                    VoxelBin(
                        r0=r0,
                        g0=g0,
                        b0=b0,
                        dr=r1 - r0,
                        dg=g1 - g0,
                        db=b1 - b0,
                        r_center=r_center,
                        g_center=g_center,
                        b_center=b_center,
                        count=count,
                        bin_color=_bin_color(r_center, g_center, b_center),
                        opacity=_display_opacity(count, mode),
                    )
                )

    return voxels or None


def _voxel_from_index(
    hist: np.ndarray,
    bin_edges: list[np.ndarray],
    ib: int,
    ig: int,
    ir: int,
) -> VoxelBin | None:
    count = float(hist[ib, ig, ir])
    if count <= 0:
        return None

    mode = float(hist.max())
    b_edges, g_edges, r_edges = bin_edges
    r0, r1 = float(r_edges[ir]), float(r_edges[ir + 1])
    g0, g1 = float(g_edges[ig]), float(g_edges[ig + 1])
    b0, b1 = float(b_edges[ib]), float(b_edges[ib + 1])
    r_center = (r0 + r1) * 0.5
    g_center = (g0 + g1) * 0.5
    b_center = (b0 + b1) * 0.5

    return VoxelBin(
        r0=r0,
        g0=g0,
        b0=b0,
        dr=r1 - r0,
        dg=g1 - g0,
        db=b1 - b0,
        r_center=r_center,
        g_center=g_center,
        b_center=b_center,
        count=count,
        bin_color=_bin_color(r_center, g_center, b_center),
        opacity=_display_opacity(count, mode),
    )


def _top_k_voxels(
    hist: np.ndarray,
    bin_edges: list[np.ndarray],
    k: int,
) -> list[VoxelBin]:
    flat = hist.ravel()
    if flat.max() <= 0:
        return []

    order = np.argsort(flat)[::-1]
    voxels: list[VoxelBin] = []
    for idx in order:
        if flat[idx] <= 0:
            break
        ib, ig, ir = np.unravel_index(int(idx), hist.shape)
        voxel = _voxel_from_index(hist, bin_edges, ib, ig, ir)
        if voxel is not None:
            voxels.append(voxel)
        if len(voxels) >= k:
            break
    return voxels


@dataclass(frozen=True)
class SubCube:
    src_center: tuple[float, float, float]
    dst_center: tuple[float, float, float]
    size: tuple[float, float, float]
    src_color: tuple[float, float, float]
    dst_color: tuple[float, float, float]
    mass: float
    move_cost: float
    src_idx: int
    dst_idx: int


def _lerp_rgb(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple((1.0 - t) * c0 + t * c1 for c0, c1 in zip(a, b))


def _subdivide_voxel_grid(voxel: VoxelBin, grid: int, mass: float) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Return (center, size) for each sub-cube in a grid×grid×grid layout inside voxel."""
    if grid <= 0:
        raise ValueError("grid must be positive.")

    dr = voxel.dr / grid
    dg = voxel.dg / grid
    db = voxel.db / grid
    size = (dr, dg, db)
    centers: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    for kb in range(grid):
        for kg in range(grid):
            for kr in range(grid):
                r_center = voxel.r0 + (kr + 0.5) * dr
                g_center = voxel.g0 + (kg + 0.5) * dg
                b_center = voxel.b0 + (kb + 0.5) * db
                centers.append(((r_center, g_center, b_center), size))

    return centers


def _allocate_destinations_for_source(
    flow_row: np.ndarray,
    source_mass: float,
    num_cells: int,
) -> list[int]:
    """Assign each sub-cube to a destination using proportional largest-remainder counts."""
    if num_cells <= 0:
        return []

    n_dest = len(flow_row)
    if source_mass <= 1e-12:
        dominant = int(np.argmax(flow_row))
        return [dominant] * num_cells

    quotas = np.array([flow_row[j] / source_mass for j in range(n_dest)], dtype=float) * num_cells
    counts = np.floor(quotas).astype(int)

    positive = [j for j in range(n_dest) if flow_row[j] > 1e-9]
    if positive and num_cells >= len(positive):
        for j in positive:
            if counts[j] == 0:
                counts[j] = 1

    while int(counts.sum()) > num_cells:
        j = int(np.argmax(counts))
        if counts[j] > 0:
            counts[j] -= 1

    remainder = num_cells - int(counts.sum())
    if remainder > 0:
        fractional = quotas - np.floor(quotas)
        order = np.argsort(-fractional)
        for j in order:
            if remainder <= 0:
                break
            counts[j] += 1
            remainder -= 1

    allocations: list[int] = []
    for j, count in enumerate(counts):
        allocations.extend([j] * int(count))

    while len(allocations) < num_cells:
        allocations.append(int(np.argmax(flow_row)))
    while len(allocations) > num_cells:
        allocations.pop()

    return allocations


def _interleave_destination_labels(labels: list[int]) -> list[int]:
    """Spread destinations across a source bin instead of grouping them in blocks."""
    if not labels:
        return []

    from collections import Counter, deque

    buckets: dict[int, deque[int]] = {
        dest: deque([dest] * count) for dest, count in Counter(labels).items()
    }
    order = sorted(buckets.keys())
    interleaved: list[int] = []
    while any(buckets[dest] for dest in order):
        for dest in order:
            if buckets[dest]:
                interleaved.append(buckets[dest].popleft())
    return interleaved


def _interleave_moves(moves: list[SubCube]) -> list[SubCube]:
    """Animate in round-robin order across source→destination pairs."""
    if not moves:
        return []

    from collections import defaultdict, deque

    buckets: dict[tuple[int, int], deque[SubCube]] = defaultdict(deque)
    for move in moves:
        buckets[(move.src_idx, move.dst_idx)].append(move)

    keys = sorted(buckets.keys())
    interleaved: list[SubCube] = []
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                interleaved.append(buckets[key].popleft())
    return interleaved


def _rgb_ground_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    metric: str,
) -> float:
    delta = np.array(b, dtype=float) - np.array(a, dtype=float)
    if metric == "euclidean":
        return float(np.linalg.norm(delta))
    if metric == "manhattan":
        return float(np.sum(np.abs(delta)))
    raise ValueError(f"Unknown EMD metric: {metric}")


def _compute_emd_flow(
    supplies: np.ndarray,
    demands: np.ndarray,
    cost: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve optimal transport between two small discrete distributions."""
    supplies = np.asarray(supplies, dtype=float)
    demands = np.asarray(demands, dtype=float)
    cost = np.asarray(cost, dtype=float)

    if supplies.ndim != 1 or demands.ndim != 1 or cost.shape != (len(supplies), len(demands)):
        raise ValueError("Invalid EMD shapes.")

    total_supply = float(supplies.sum())
    total_demand = float(demands.sum())
    if total_supply <= 0 or total_demand <= 0:
        raise ValueError("EMD requires positive total mass in both histograms.")

    supplies = supplies / total_supply
    demands = demands / total_demand

    n, m = len(supplies), len(demands)
    if n == 2 and m == 2:
        s0, s1 = supplies
        d0, d1 = demands
        lo = max(0.0, s0 - d1)
        hi = min(s0, d0)
        best_flow = np.zeros((2, 2), dtype=float)
        best_cost = float("inf")
        for f00 in (lo, hi):
            f01 = s0 - f00
            f10 = d0 - f00
            f11 = d1 - f01
            if min(f01, f10, f11) < -1e-12:
                continue
            flow = np.array([[f00, f01], [f10, f11]], dtype=float)
            total = float(np.sum(flow * cost))
            if total < best_cost:
                best_cost = total
                best_flow = flow
        if not np.isfinite(best_cost):
            raise RuntimeError("EMD solver failed to find a feasible 2×2 transport plan.")
        return best_flow, best_cost

    # Fallback for larger tiny problems: enumerate extreme points via northwest-corner pivots.
    flow = np.zeros((n, m), dtype=float)
    remaining_supply = supplies.copy()
    remaining_demand = demands.copy()
    i = 0
    j = 0
    while i < n and j < m:
        amount = min(remaining_supply[i], remaining_demand[j])
        flow[i, j] = amount
        remaining_supply[i] -= amount
        remaining_demand[j] -= amount
        if remaining_supply[i] <= 1e-12:
            i += 1
        if remaining_demand[j] <= 1e-12:
            j += 1

    # Northwest-corner is not always optimal; refine with pairwise swaps when small.
    improved = True
    while improved:
        improved = False
        for i0 in range(n):
            for j0 in range(m):
                for i1 in range(n):
                    for j1 in range(m):
                        if i0 == i1 and j0 == j1:
                            continue
                        delta = min(flow[i0, j0], flow[i1, j1])
                        if delta <= 1e-12:
                            continue
                        before = (
                            delta * cost[i0, j0]
                            + delta * cost[i1, j1]
                        )
                        after = (
                            delta * cost[i0, j1]
                            + delta * cost[i1, j0]
                        )
                        if after + 1e-12 < before:
                            flow[i0, j0] -= delta
                            flow[i1, j1] -= delta
                            flow[i0, j1] += delta
                            flow[i1, j0] += delta
                            improved = True

    emd_score = float(np.sum(flow * cost))
    return flow, emd_score


def _build_subcube_moves(
    source_voxels: list[VoxelBin],
    target_voxels: list[VoxelBin],
    flow: np.ndarray,
    supplies: np.ndarray,
    grid: int,
    metric: str,
) -> tuple[list[SubCube], float]:
    source_grids = [
        _subdivide_voxel_grid(voxel, grid, float(voxel.count))
        for voxel in source_voxels
    ]
    target_grids = [
        _subdivide_voxel_grid(voxel, grid, float(voxel.count))
        for voxel in target_voxels
    ]

    planned: list[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            int,
            int,
        ]
    ] = []
    dest_cursor = [0] * len(target_voxels)

    for src_idx, (src_voxel, src_cells) in enumerate(zip(source_voxels, source_grids)):
        source_mass = float(supplies[src_idx])
        dest_allocations = _allocate_destinations_for_source(
            flow[src_idx], source_mass, len(src_cells)
        )
        dest_allocations = _interleave_destination_labels(dest_allocations)

        for (src_center, size), dst_idx in zip(src_cells, dest_allocations):
            dst_cells = target_grids[dst_idx]
            dst_slot = dest_cursor[dst_idx] % len(dst_cells)
            dst_center, _ = dst_cells[dst_slot]
            dest_cursor[dst_idx] += 1
            planned.append(
                (src_center, dst_center, size, src_voxel.bin_color, target_voxels[dst_idx].bin_color, src_idx, dst_idx)
            )

    pair_counts: dict[tuple[int, int], int] = {}
    for *_, src_idx, dst_idx in planned:
        key = (src_idx, dst_idx)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    moves: list[SubCube] = []
    total_cost = 0.0
    for src_center, dst_center, size, src_color, dst_color, src_idx, dst_idx in planned:
        src_voxel = source_voxels[src_idx]
        dst_voxel = target_voxels[dst_idx]
        dist = _rgb_ground_distance(
            (src_voxel.r_center, src_voxel.g_center, src_voxel.b_center),
            (dst_voxel.r_center, dst_voxel.g_center, dst_voxel.b_center),
            metric,
        )
        pair_cost = float(flow[src_idx, dst_idx]) * dist
        pair_size = pair_counts[(src_idx, dst_idx)]
        move_cost = pair_cost / pair_size if pair_size > 0 else 0.0
        sub_mass = float(src_voxel.count) / (grid ** 3)
        total_cost += move_cost
        moves.append(
            SubCube(
                src_center=src_center,
                dst_center=dst_center,
                size=size,
                src_color=src_color,
                dst_color=dst_color,
                mass=sub_mass,
                move_cost=move_cost,
                src_idx=src_idx,
                dst_idx=dst_idx,
            )
        )

    moves = _interleave_moves(moves)
    return moves, total_cost


def _voxel_size(voxel: VoxelBin) -> tuple[float, float, float]:
    return (voxel.dr, voxel.dg, voxel.db)


def _emd_bin_fill_alpha(fill_fraction: float, count: float, mode: float) -> float:
    weight = _display_opacity(count, mode)
    frac = float(np.clip(fill_fraction, 0.0, 1.0))
    return float(np.clip(0.1 + 0.75 * frac * weight, 0.0, 1.0))


def _build_bin_flow_moves(
    source_voxels: list[VoxelBin],
    target_voxels: list[VoxelBin],
    flow: np.ndarray,
    supplies: np.ndarray,
    metric: str,
) -> tuple[list[SubCube], float]:
    """One animated move per source→target pair with positive optimal flow."""
    raw_total = float(np.sum(supplies))
    moves: list[SubCube] = []
    total_cost = 0.0

    for src_idx, src_voxel in enumerate(source_voxels):
        source_mass = float(supplies[src_idx])
        src_center = (src_voxel.r_center, src_voxel.g_center, src_voxel.b_center)
        for dst_idx, dst_voxel in enumerate(target_voxels):
            flow_ij = float(flow[src_idx, dst_idx])
            if flow_ij <= 1e-9:
                continue

            dst_center = (dst_voxel.r_center, dst_voxel.g_center, dst_voxel.b_center)
            dist = _rgb_ground_distance(src_center, dst_center, metric)
            move_cost = flow_ij * dist
            total_cost += move_cost

            size = _voxel_size(src_voxel)
            moves.append(
                SubCube(
                    src_center=src_center,
                    dst_center=dst_center,
                    size=size,
                    src_color=src_voxel.bin_color,
                    dst_color=dst_voxel.bin_color,
                    mass=flow_ij * raw_total,
                    move_cost=move_cost,
                    src_idx=src_idx,
                    dst_idx=dst_idx,
                )
            )

    moves = _interleave_moves(moves)
    return moves, total_cost


def _draw_voxel_fill(
    ax,
    voxel: VoxelBin,
    color: tuple[float, float, float],
    *,
    alpha: float,
    edgecolor: str | None = None,
) -> None:
    ax.bar3d(
        voxel.r0,
        voxel.g0,
        voxel.b0,
        voxel.dr,
        voxel.dg,
        voxel.db,
        color=[color],
        edgecolor=edgecolor if edgecolor is not None else "none",
        linewidth=0.0,
        shade=False,
        alpha=alpha,
    )


def _compute_emd_bin_mass_state(
    moves: list[SubCube],
    initial_source: np.ndarray,
    initial_dest: np.ndarray,
    completed: int,
    progress: float,
) -> tuple[np.ndarray, np.ndarray]:
    remaining = initial_source.astype(float).copy()
    arrived = np.zeros(len(initial_dest), dtype=float)
    for idx, move in enumerate(moves):
        if idx < completed:
            remaining[move.src_idx] -= move.mass
            arrived[move.dst_idx] += move.mass
        elif idx == completed and progress > 0.0:
            remaining[move.src_idx] -= move.mass * progress
    return np.clip(remaining, 0.0, None), np.clip(arrived, 0.0, None)


def _format_emd_count(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}"


def _emd_bin_count_caption(
    source_voxels: list[VoxelBin],
    target_voxels: list[VoxelBin],
    remaining_source: np.ndarray,
    arrived_dest: np.ndarray,
) -> str:
    src_parts = [
        f"S{i + 1} {_format_emd_count(remaining_source[i])}/{_format_emd_count(source_voxels[i].count)}"
        for i in range(len(source_voxels))
    ]
    dst_parts = [
        f"T{j + 1} {_format_emd_count(arrived_dest[j])}/{_format_emd_count(target_voxels[j].count)}"
        for j in range(len(target_voxels))
    ]
    return "Sources left: " + ", ".join(src_parts) + " | Targets filled: " + ", ".join(dst_parts)


def _draw_subcube(ax, center: tuple[float, float, float], size: tuple[float, float, float], color, *, alpha: float, edgecolor: str = "black") -> None:
    r, g, b = center
    dr, dg, db = size
    ax.bar3d(
        r - dr * 0.5,
        g - dg * 0.5,
        b - db * 0.5,
        dr,
        dg,
        db,
        color=[color],
        edgecolor=edgecolor,
        linewidth=0.4,
        shade=False,
        alpha=alpha,
    )


def _draw_voxel_outline(ax, voxel: VoxelBin, *, color: str, alpha: float, linewidth: float) -> None:
    x0, x1 = voxel.r0, voxel.r0 + voxel.dr
    y0, y1 = voxel.g0, voxel.g0 + voxel.dg
    z0, z1 = voxel.b0, voxel.b0 + voxel.db
    corners = (
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    )
    edge_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for a, b in edge_pairs:
        ax.plot(
            (corners[a][0], corners[b][0]),
            (corners[a][1], corners[b][1]),
            (corners[a][2], corners[b][2]),
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


def visualize_emd_transport(
    image_a: Path,
    image_b: Path,
    hist_a: np.ndarray,
    hist_b: np.ndarray,
    bin_edges: list[np.ndarray],
    *,
    output_path: Path,
    grid: int,
    fps: int,
    top_k: int,
    metric: str,
    bins_only: bool,
    show: bool,
) -> None:
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    source_voxels = _top_k_voxels(hist_a, bin_edges, top_k)
    target_voxels = _top_k_voxels(hist_b, bin_edges, top_k)
    if len(source_voxels) < top_k or len(target_voxels) < top_k:
        raise RuntimeError(
            f"Each histogram needs at least {top_k} occupied bins for EMD visualization "
            f"(found {len(source_voxels)} and {len(target_voxels)})."
        )

    supplies = np.array([v.count for v in source_voxels], dtype=float)
    demands = np.array([v.count for v in target_voxels], dtype=float)
    src_centers = [(v.r_center, v.g_center, v.b_center) for v in source_voxels]
    dst_centers = [(v.r_center, v.g_center, v.b_center) for v in target_voxels]
    cost = np.array(
        [[_rgb_ground_distance(s, d, metric) for d in dst_centers] for s in src_centers],
        dtype=float,
    )
    flow, emd_score = _compute_emd_flow(supplies, demands, cost)
    supplies_norm = supplies / supplies.sum()
    if bins_only:
        moves, animated_cost = _build_bin_flow_moves(
            source_voxels, target_voxels, flow, supplies, metric
        )
    else:
        moves, animated_cost = _build_subcube_moves(
            source_voxels, target_voxels, flow, supplies_norm, grid, metric
        )

    initial_source = np.array([v.count for v in source_voxels], dtype=float)
    initial_dest = np.array([v.count for v in target_voxels], dtype=float)
    source_mode = float(initial_source.max()) if initial_source.size else 1.0
    dest_mode = float(initial_dest.max()) if initial_dest.size else 1.0

    metric_label = "L2 (Euclidean)" if metric == "euclidean" else "L1 (Manhattan)"
    print(f"EMD between top-{top_k} bins ({image_a.name} → {image_b.name}, {metric_label}): {emd_score:.4f}")
    print("Source bin counts:", ", ".join(_format_emd_count(v.count) for v in source_voxels))
    print("Target bin counts:", ", ".join(_format_emd_count(v.count) for v in target_voxels))
    print("Optimal flow (source bin → target bin):")
    for i, src in enumerate(source_voxels):
        for j, dst in enumerate(target_voxels):
            if flow[i, j] > 1e-6:
                print(
                    f"  bin {i + 1} (RGB≈{src.r_center:.0f},{src.g_center:.0f},{src.b_center:.0f}) "
                    f"→ bin {j + 1} (RGB≈{dst.r_center:.0f},{dst.g_center:.0f},{dst.b_center:.0f}): "
                    f"flow={flow[i, j]:.4f}"
                )
    if bins_only:
        print(f"Animating {len(moves)} bin-level flow moves (no sub-cubes).")
    else:
        print(f"Animating {len(moves)} sub-cube moves on a {grid}×{grid}×{grid} grid per bin.")

    frames_per_move = 5
    fig = plt.figure(figsize=(11, 8))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    writer.setup(fig, str(output_path), dpi=fig.dpi)

    def render_frame(completed: int, progress: float, running_cost: float):
        fig.clear()
        ax = fig.add_subplot(111, projection="3d")
        _draw_outer_cube_wireframe(ax)

        remaining_source, arrived_dest = _compute_emd_bin_mass_state(
            moves, initial_source, initial_dest, completed, progress
        )

        for j, voxel in enumerate(target_voxels):
            demand_frac = _display_opacity(float(initial_dest[j]), dest_mode)
            arrived_frac = float(
                np.clip(
                    arrived_dest[j] / initial_dest[j] if initial_dest[j] > 0 else 0.0,
                    0.0,
                    1.0,
                )
            )
            _draw_voxel_outline(
                ax,
                voxel,
                color="#1f77b4",
                alpha=0.2 + 0.55 * demand_frac,
                linewidth=1.2,
            )
            if arrived_frac > 1e-3:
                fill_alpha = _emd_bin_fill_alpha(arrived_frac, float(initial_dest[j]), dest_mode)
                _draw_voxel_fill(ax, voxel, voxel.bin_color, alpha=fill_alpha)

        for i, voxel in enumerate(source_voxels):
            supply_frac = _display_opacity(float(initial_source[i]), source_mode)
            remaining_frac = float(
                np.clip(
                    remaining_source[i] / initial_source[i] if initial_source[i] > 0 else 0.0,
                    0.0,
                    1.0,
                )
            )
            if remaining_frac > 1e-3:
                fill_alpha = _emd_bin_fill_alpha(remaining_frac, float(initial_source[i]), source_mode)
                _draw_voxel_fill(ax, voxel, voxel.bin_color, alpha=fill_alpha)
            _draw_voxel_outline(
                ax,
                voxel,
                color="#d62728",
                alpha=float(np.clip(0.25 + 0.65 * remaining_frac * supply_frac, 0.0, 1.0)),
                linewidth=1.0 + 1.5 * remaining_frac,
            )

        for idx, move in enumerate(moves):
            if bins_only:
                if idx == completed and progress > 0.0:
                    t = progress
                    interp = tuple(
                        (1.0 - t) * s + t * d
                        for s, d in zip(move.src_center, move.dst_center)
                    )
                    interp_color = _lerp_rgb(move.src_color, move.dst_color, t)
                    src_alpha = _emd_bin_fill_alpha(
                        1.0, float(initial_source[move.src_idx]), source_mode
                    )
                    future_arrived = arrived_dest[move.dst_idx] + move.mass
                    future_frac = float(
                        np.clip(
                            future_arrived / initial_dest[move.dst_idx]
                            if initial_dest[move.dst_idx] > 0
                            else 0.0,
                            0.0,
                            1.0,
                        )
                    )
                    dst_alpha = _emd_bin_fill_alpha(
                        future_frac, float(initial_dest[move.dst_idx]), dest_mode
                    )
                    blob_alpha = float(np.clip((1.0 - t) * src_alpha + t * dst_alpha, 0.0, 1.0))
                    _draw_subcube(
                        ax,
                        interp,
                        move.size,
                        interp_color,
                        alpha=blob_alpha,
                        edgecolor="#ff7f0e",
                    )
                continue

            src_remaining_frac = float(
                np.clip(
                    remaining_source[move.src_idx] / initial_source[move.src_idx]
                    if initial_source[move.src_idx] > 0
                    else 0.0,
                    0.0,
                    1.0,
                )
            )
            dst_arrived_frac = float(
                np.clip(
                    arrived_dest[move.dst_idx] / initial_dest[move.dst_idx]
                    if initial_dest[move.dst_idx] > 0
                    else 0.0,
                    0.0,
                    1.0,
                )
            )
            if idx < completed:
                cube_alpha = float(np.clip(0.35 + 0.6 * dst_arrived_frac, 0.0, 1.0))
                _draw_subcube(ax, move.dst_center, move.size, move.dst_color, alpha=cube_alpha)
            elif idx == completed and progress > 0.0:
                t = progress
                interp = tuple(
                    (1.0 - t) * s + t * d
                    for s, d in zip(move.src_center, move.dst_center)
                )
                interp_color = _lerp_rgb(move.src_color, move.dst_color, t)
                _draw_subcube(ax, interp, move.size, interp_color, alpha=0.95, edgecolor="#ff7f0e")
            else:
                cube_alpha = float(np.clip(0.25 + 0.65 * src_remaining_frac, 0.0, 1.0))
                _draw_subcube(ax, move.src_center, move.size, move.src_color, alpha=cube_alpha)

        _apply_bin_axis_ticks_matplotlib(ax, bin_edges)
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        ax.set_xlim(float(bin_edges[2][0]), float(bin_edges[2][-1]))
        ax.set_ylim(float(bin_edges[1][0]), float(bin_edges[1][-1]))
        ax.set_zlim(float(bin_edges[0][0]), float(bin_edges[0][-1]))
        ax.view_init(elev=24, azim=-58)
        move_label = (
            f"Flow {min(completed + 1, len(moves))}/{len(moves)}"
            if bins_only
            else f"Move {min(completed + 1, len(moves))}/{len(moves)} ({grid}×{grid}×{grid} sub-cubes per bin)"
        )
        ax.set_title(
            f"Earth Mover's Distance — top-{top_k} bins ({metric_label})\n"
            f"{image_a.name} → {image_b.name} | "
            f"EMD: {running_cost:.4f} / {emd_score:.4f}\n"
            f"{move_label}\n"
            f"{_emd_bin_count_caption(source_voxels, target_voxels, remaining_source, arrived_dest)}",
            fontsize=10,
        )
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
        return ax

    running_cost = 0.0
    render_frame(0, 0.0, running_cost)
    fig.canvas.draw()
    writer.grab_frame()

    for move_idx, move in enumerate(moves):
        for step in range(1, frames_per_move + 1):
            progress = step / frames_per_move
            render_frame(move_idx, progress, running_cost + move.move_cost * progress)
            fig.canvas.draw()
            writer.grab_frame()
        running_cost += move.move_cost

    render_frame(len(moves), 0.0, emd_score)
    fig.canvas.draw()
    for _ in range(fps):
        writer.grab_frame()
    writer.finish()

    print(f"Saved EMD animation to {output_path}")
    print(f"Animated transport cost tally: {animated_cost:.4f} (solver EMD: {emd_score:.4f})")

    if show:
        os.environ.pop("MPLBACKEND", None)
        plt.show()
    else:
        plt.close(fig)


def _format_bin_range_label(lo: float, hi: float) -> str:
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i > lo_i:
        hi_i -= 1
    return f"{lo_i}–{hi_i}"


def _axis_bin_ticks(edges: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Tick at bin centers with labels showing each bin's pixel value range."""
    positions = (edges[:-1] + edges[1:]) / 2
    labels = [_format_bin_range_label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return positions, labels


def _apply_bin_axis_ticks_matplotlib(ax, bin_edges: list[np.ndarray]) -> None:
    b_edges, g_edges, r_edges = bin_edges
    r_pos, r_labels = _axis_bin_ticks(r_edges)
    g_pos, g_labels = _axis_bin_ticks(g_edges)
    b_pos, b_labels = _axis_bin_ticks(b_edges)

    ax.set_xticks(r_pos)
    ax.set_xticklabels(r_labels, fontsize=8)
    ax.set_yticks(g_pos)
    ax.set_yticklabels(g_labels, fontsize=8)
    ax.set_zticks(b_pos)
    ax.set_zticklabels(b_labels, fontsize=8)


def _plotly_bin_axis(edges: np.ndarray, title: str) -> dict:
    positions, labels = _axis_bin_ticks(edges)
    return dict(
        title=title,
        range=[float(edges[0]), float(edges[-1])],
        tickmode="array",
        tickvals=positions.tolist(),
        ticktext=labels,
        showbackground=False,
    )


def _format_bin_count_label(count: float, normalized: bool, num_pixels: int) -> str:
    if normalized and num_pixels > 0:
        return str(int(round(count * num_pixels)))
    if float(count).is_integer():
        return str(int(count))
    return f"{count:.3f}"


def _attach_billboard_labels(
    ax,
    voxels: list[VoxelBin],
    *,
    normalized: bool,
    num_pixels: int,
) -> None:
    """2D labels projected from 3D bin centers; stay upright with white boxes while rotating."""
    from matplotlib.text import Annotation
    from mpl_toolkits.mplot3d import proj3d

    label_bbox = dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        alpha=1.0,
    )

    class Annotation3D(Annotation):
        def __init__(self, text: str, xyz: tuple[float, float, float], **kwargs) -> None:
            super().__init__(
                text,
                xy=(0.0, 0.0),
                xytext=(0.0, 0.0),
                xycoords="data",
                textcoords="offset points",
                **kwargs,
            )
            self.xyz = xyz

        def draw(self, renderer) -> None:
            x3, y3, z3 = self.xyz
            x2, y2, _ = proj3d.proj_transform(x3, y3, z3, self.axes.get_proj())
            self.xy = (x2, y2)
            self.set_position((x2, y2))
            self.textcoords = "data"
            super().draw(renderer)

    for voxel in voxels:
        label = Annotation3D(
            _format_bin_count_label(voxel.count, normalized, num_pixels),
            (voxel.r_center, voxel.g_center, voxel.b_center),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            bbox=label_bbox,
            annotation_clip=False,
            zorder=1000,
        )
        ax.add_artist(label)


def _draw_outer_cube_wireframe(ax, size: float = 255.0) -> None:
    corners = [
        (0, 0, 0),
        (size, 0, 0),
        (size, size, 0),
        (0, size, 0),
        (0, 0, size),
        (size, 0, size),
        (size, size, size),
        (0, size, size),
    ]
    edge_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for a, b in edge_pairs:
        xs = (corners[a][0], corners[b][0])
        ys = (corners[a][1], corners[b][1])
        zs = (corners[a][2], corners[b][2])
        ax.plot(xs, ys, zs, color="black", linewidth=1.5, alpha=0.9)


def plot_3d_histogram_matplotlib(
    hist: np.ndarray,
    bin_edges: list[np.ndarray],
    *,
    title: str,
    output_path: Path | None,
    show: bool,
    min_fraction: float,
    normalized: bool,
    num_pixels: int,
    show_counts: bool,
) -> None:
    import os

    if show:
        os.environ.pop("MPLBACKEND", None)
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    voxels = _extract_voxels(hist, bin_edges, min_fraction)
    if voxels is None:
        print("No occupied bins to plot; try lowering --min-fraction.", file=sys.stderr)
        return

    r0 = np.array([v.r0 for v in voxels], dtype=float)
    g0 = np.array([v.g0 for v in voxels], dtype=float)
    b0 = np.array([v.b0 for v in voxels], dtype=float)
    dr = np.array([v.dr for v in voxels], dtype=float)
    dg = np.array([v.dg for v in voxels], dtype=float)
    db = np.array([v.db for v in voxels], dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for voxel, r, g, b, d_r, d_g, d_b in zip(voxels, r0, g0, b0, dr, dg, db):
        ax.bar3d(
            r,
            g,
            b,
            d_r,
            d_g,
            d_b,
            color=[voxel.bin_color],
            edgecolor="black",
            linewidth=0.6,
            shade=False,
            alpha=voxel.opacity,
        )
    _draw_outer_cube_wireframe(ax)
    if show_counts:
        _attach_billboard_labels(
            ax,
            voxels,
            normalized=normalized,
            num_pixels=num_pixels,
        )

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_xlim(float(bin_edges[2][0]), float(bin_edges[2][-1]))
    ax.set_ylim(float(bin_edges[1][0]), float(bin_edges[1][-1]))
    ax.set_zlim(float(bin_edges[0][0]), float(bin_edges[0][-1]))
    _apply_bin_axis_ticks_matplotlib(ax, bin_edges)
    ax.set_title(title + f"\n({len(voxels)} occupied bins; opacity ∝ count / mode)")
    ax.view_init(elev=22, azim=-60)
    plt.tight_layout()
    fig.canvas.draw()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")

    if show:
        print("Close the plot window to exit.")
        plt.show()
    else:
        plt.close(fig)


def _cube_mesh_vertices(voxel: VoxelBin) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    x0, x1 = voxel.r0, voxel.r0 + voxel.dr
    y0, y1 = voxel.g0, voxel.g0 + voxel.dg
    z0, z1 = voxel.b0, voxel.b0 + voxel.db
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    faces = (
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (2, 6, 7), (2, 7, 3),
        (0, 3, 7), (0, 7, 4),
        (1, 5, 6), (1, 6, 2),
    )
    i, j, k = zip(*faces)
    return x, y, z, list(i), list(j), list(k)


def _cube_edge_lines(voxel: VoxelBin) -> tuple[list[float], list[float], list[float]]:
    x0, x1 = voxel.r0, voxel.r0 + voxel.dr
    y0, y1 = voxel.g0, voxel.g0 + voxel.dg
    z0, z1 = voxel.b0, voxel.b0 + voxel.db
    corners = (
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    )
    edge_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for a, b in edge_pairs:
        xs.extend((corners[a][0], corners[b][0], None))
        ys.extend((corners[a][1], corners[b][1], None))
        zs.extend((corners[a][2], corners[b][2], None))
    return xs, ys, zs


def _outer_cube_edge_lines(size: float = 255.0) -> tuple[list[float | None], list[float | None], list[float | None]]:
    corners = [
        (0, 0, 0),
        (size, 0, 0),
        (size, size, 0),
        (0, size, 0),
        (0, 0, size),
        (size, 0, size),
        (size, size, size),
        (0, size, size),
    ]
    edge_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for a, b in edge_pairs:
        xs.extend((corners[a][0], corners[b][0], None))
        ys.extend((corners[a][1], corners[b][1], None))
        zs.extend((corners[a][2], corners[b][2], None))
    return xs, ys, zs


def _rgb_to_hex(color: tuple[float, float, float]) -> str:
    r, g, b = (np.array(color) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_3d_histogram_html(
    hist: np.ndarray,
    bin_edges: list[np.ndarray],
    *,
    title: str,
    output_path: Path,
    min_fraction: float,
    normalized: bool,
    num_pixels: int,
    show_counts: bool,
) -> None:
    """Write a rotatable voxel histogram as an interactive HTML file."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install plotly for HTML output: pip install plotly") from exc

    voxels = _extract_voxels(hist, bin_edges, min_fraction)
    if voxels is None:
        print("No occupied bins to plot; try lowering --min-fraction.", file=sys.stderr)
        return

    traces = []
    for voxel in voxels:
        x, y, z, i, j, k = _cube_mesh_vertices(voxel)
        fill_color = _rgb_to_hex(voxel.bin_color)
        traces.append(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=fill_color,
                opacity=voxel.opacity,
                flatshading=True,
                hovertemplate=(
                    f"R={voxel.r_center:.0f}<br>"
                    f"G={voxel.g_center:.0f}<br>"
                    f"B={voxel.b_center:.0f}<br>"
                    f"count={voxel.count:.4f}<extra></extra>"
                ),
                showscale=False,
            )
        )
        ex, ey, ez = _cube_edge_lines(voxel)
        traces.append(
            go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color="black", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    ox, oy, oz = _outer_cube_edge_lines()
    traces.append(
        go.Scatter3d(
            x=ox,
            y=oy,
            z=oz,
            mode="lines",
            line=dict(color="black", width=5),
            hoverinfo="skip",
            showlegend=False,
            name="RGB cube",
        )
    )

    annotations = []
    if show_counts:
        annotations = [
            dict(
                x=voxel.r_center,
                y=voxel.g_center,
                z=voxel.b_center,
                text=_format_bin_count_label(voxel.count, normalized, num_pixels),
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                opacity=1.0,
                font=dict(color="black", size=14),
            )
            for voxel in voxels
        ]

    b_edges, g_edges, r_edges = bin_edges
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title}<br><sup>{len(voxels)} occupied bins; opacity ∝ count / mode</sup>",
        scene=dict(
            annotations=annotations,
            xaxis=_plotly_bin_axis(r_edges, "Red"),
            yaxis=_plotly_bin_axis(g_edges, "Green"),
            zaxis=_plotly_bin_axis(b_edges, "Blue"),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved interactive 3D plot to {output_path}")
    print("Open the HTML file in a browser and drag to rotate.")


def save_histogram_data(hist: np.ndarray, output_path: Path) -> None:
    np.save(output_path, hist)
    print(f"Saved histogram array to {output_path}")


def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def save_grayscale_image(image_bgr: np.ndarray, output_path: Path) -> np.ndarray:
    gray = to_grayscale(image_bgr)
    cv2.imwrite(str(output_path), gray)
    print(f"Saved grayscale image to {output_path}")
    return gray


def compute_1d_histogram(
    channel: np.ndarray, num_bins: int, normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, 256, num_bins + 1)
    hist, _ = np.histogram(channel.ravel(), bins=edges)
    hist = hist.astype(np.float64)
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()
    return hist, edges


def compute_rgb_1d_histograms(
    image_bgr: np.ndarray, num_bins: int, normalize: bool
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    b, g, r = cv2.split(image_bgr)
    edges = np.linspace(0, 256, num_bins + 1)
    hists: dict[str, np.ndarray] = {}
    for name, channel in (("Blue", b), ("Green", g), ("Red", r)):
        hist, _ = np.histogram(channel.ravel(), bins=edges)
        hist = hist.astype(np.float64)
        if normalize and hist.sum() > 0:
            hist = hist / hist.sum()
        hists[name] = hist
    return hists, edges


def _display_channel_bar_color(
    channel: str, center: float, count: float, mode: float
) -> tuple[float, float, float]:
    ratio = _frequency_ratio(count, mode)
    neutral = 0.88
    blend = 0.12 + 0.88 * ratio
    value = center / 255.0
    if channel == "Red":
        full = (value, 0.0, 0.0)
    elif channel == "Green":
        full = (0.0, value, 0.0)
    else:
        full = (0.0, 0.0, value)
    neutral_rgb = (neutral, neutral, neutral)
    return tuple(neutral_rgb[i] + blend * (full[i] - neutral_rgb[i]) for i in range(3))


def _channel_bar_colors(
    channel: str, hist: np.ndarray, edges: np.ndarray
) -> list[tuple[float, float, float]]:
    mode = float(hist.max())
    return [
        _display_channel_bar_color(channel, (lo + hi) * 0.5, count, mode)
        for count, lo, hi in zip(hist, edges[:-1], edges[1:])
    ]


def _ylabel_for_hist(hist: np.ndarray, normalized: bool) -> str:
    if normalized:
        return "Normalized count"
    return "Count"


def _draw_1d_hist_on_axes(
    ax,
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    title: str,
    bar_colors: list[tuple[float, float, float]],
    normalized: bool,
) -> None:
    positions, labels = _axis_bin_ticks(edges)
    ax.bar(
        positions,
        hist,
        width=(edges[1] - edges[0]) * 0.9,
        align="center",
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Intensity range")
    ax.set_ylabel(_ylabel_for_hist(hist, normalized))
    ax.set_title(title)
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.grid(axis="y", alpha=0.3)


def plot_rgb_1d_histograms_matplotlib(
    hists: dict[str, np.ndarray],
    edges: np.ndarray,
    *,
    image_name: str,
    output_path: Path | None,
    show: bool,
    normalized: bool,
) -> None:
    import os

    if show:
        os.environ.pop("MPLBACKEND", None)
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    channel_order = ("Red", "Green", "Blue")
    fig, axes = plt.subplots(len(channel_order), 1, figsize=(10, 11), sharex=True)
    if len(channel_order) == 1:
        axes = [axes]

    for ax, channel in zip(axes, channel_order):
        hist = hists[channel]
        colors = _channel_bar_colors(channel, hist, edges)
        _draw_1d_hist_on_axes(
            ax,
            hist,
            edges,
            title=f"{channel} channel — {image_name}",
            bar_colors=colors,
            normalized=normalized,
        )

    fig.suptitle(f"RGB 1D Histograms — {image_name}", y=0.995)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved RGB 1D histograms to {output_path}")

    if show:
        print("Close the plot window to exit.")
        plt.show()
    else:
        plt.close(fig)


def plot_rgb_1d_histograms_html(
    hists: dict[str, np.ndarray],
    edges: np.ndarray,
    *,
    image_name: str,
    output_path: Path,
    normalized: bool,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Install plotly for HTML output: pip install plotly") from exc

    channel_order = ("Red", "Green", "Blue")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{channel} channel" for channel in channel_order],
        vertical_spacing=0.08,
    )

    positions, labels = _axis_bin_ticks(edges)
    y_title = "Normalized count" if normalized else "Count"
    bar_width = (edges[1] - edges[0]) * 0.9

    for row, channel in enumerate(channel_order, start=1):
        hist = hists[channel]
        colors = [_rgb_to_hex(c) for c in _channel_bar_colors(channel, hist, edges)]
        fig.add_trace(
            go.Bar(
                x=positions,
                y=hist,
                width=bar_width,
                marker=dict(color=colors, line=dict(color="black", width=1)),
                text=labels,
                hovertemplate="Range: %{text}<br>Count: %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=y_title, row=row, col=1)

    fig.update_xaxes(
        title_text="Intensity range",
        tickmode="array",
        tickvals=positions.tolist(),
        ticktext=labels,
        range=[float(edges[0]), float(edges[-1])],
        row=3,
        col=1,
    )
    fig.update_layout(
        title=f"RGB 1D Histograms — {image_name}",
        margin=dict(l=40, r=20, b=80, t=80),
        height=900,
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved interactive RGB 1D histograms to {output_path}")


def _display_gray(center: float, count: float, mode: float) -> tuple[float, float, float]:
    ratio = _frequency_ratio(count, mode)
    full = center / 255.0
    neutral = 0.88
    blend = 0.12 + 0.88 * ratio
    value = neutral + blend * (full - neutral)
    return (value, value, value)


def plot_1d_histogram_matplotlib(
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    title: str,
    output_path: Path | None,
    show: bool,
    normalized: bool,
) -> None:
    import os

    if show:
        os.environ.pop("MPLBACKEND", None)
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    mode = float(hist.max())
    colors = [
        _display_gray((lo + hi) * 0.5, count, mode)
        for count, lo, hi in zip(hist, edges[:-1], edges[1:])
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_1d_hist_on_axes(
        ax,
        hist,
        edges,
        title=title,
        bar_colors=colors,
        normalized=normalized,
    )
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved 1D histogram to {output_path}")

    if show:
        print("Close the plot window to exit.")
        plt.show()
    else:
        plt.close(fig)


def plot_1d_histogram_html(
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Install plotly for HTML output: pip install plotly") from exc

    mode = float(hist.max())
    positions, labels = _axis_bin_ticks(edges)
    colors = [
        _rgb_to_hex(_display_gray((lo + hi) * 0.5, count, mode))
        for count, lo, hi in zip(hist, edges[:-1], edges[1:])
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=positions,
                y=hist,
                width=(edges[1] - edges[0]) * 0.9,
                marker=dict(color=colors, line=dict(color="black", width=1)),
                text=labels,
                hovertemplate="Range: %{text}<br>Count: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Intensity range",
            tickmode="array",
            tickvals=positions.tolist(),
            ticktext=labels,
            range=[float(edges[0]), float(edges[-1])],
        ),
        yaxis_title="Normalized count" if hist.sum() <= 1.0001 else "Count",
        margin=dict(l=40, r=20, b=80, t=60),
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved interactive 1D histogram to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and visualize RGB 3D, grayscale 1D, or per-channel RGB 1D histograms."
    )
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Number of bins per channel (default: 8). Each bin spans 256 / bins pixel values.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=None,
        help="Width of each bin in pixel values [1-256]. Overrides --bins when set.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save a static PNG snapshot of the 3D plot",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Save a rotatable 3D plot as an interactive HTML file",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help="Save raw 3D histogram array as a .npy file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive 3D window (drag to rotate)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep raw pixel counts instead of normalizing to a probability distribution",
    )
    parser.add_argument(
        "--1d",
        dest="hist_1d",
        action="store_true",
        help="Compute a 1D grayscale histogram (converts color images to greyscale)",
    )
    parser.add_argument(
        "--rgb-1d",
        dest="hist_rgb_1d",
        action="store_true",
        help="Compute separate 1D histograms for the Red, Green, and Blue channels",
    )
    parser.add_argument(
        "--save-grayscale",
        type=Path,
        default=None,
        help="Save a greyscale version of the input image to this path",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.0,
        help="Only plot bins with count >= this fraction of the mode (default: 0 = all non-zero bins)",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        default=True,
        help="Overlay each occupied bin with its count (default: on)",
    )
    parser.add_argument(
        "--no-counts",
        dest="show_counts",
        action="store_false",
        help="Hide count labels on occupied bins",
    )
    parser.add_argument(
        "--emd",
        type=Path,
        default=None,
        metavar="IMAGE2",
        help="Visualize Earth Mover's Distance against a second image (sub-cube transport)",
    )
    parser.add_argument(
        "--emd-top-k",
        type=int,
        default=4,
        metavar="N",
        help="Use the N most prevalent bins from each histogram for EMD (default: 2)",
    )
    parser.add_argument(
        "--emd-output",
        type=Path,
        default=None,
        help="Save EMD transport animation (.gif). Required with --emd unless --show is set.",
    )
    parser.add_argument(
        "--emd-grid",
        type=int,
        default=4,
        help="With --emd-subcubes: subdivide each top bin into N×N×N sub-cubes (default: 4)",
    )
    parser.add_argument(
        "--emd-subcubes",
        action="store_true",
        help="Animate EMD with a sub-cube grid inside each bin (see --emd-grid)",
    )
    parser.add_argument(
        "--emd-fps",
        type=int,
        default=4,
        help="Frames per second for the EMD animation (default: 8)",
    )
    parser.add_argument(
        "--emd-metric",
        choices=("euclidean", "manhattan"),
        default="euclidean",
        help="Ground distance between RGB bin centers for EMD (default: euclidean)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.image.is_file():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 1

    try:
        num_bins = resolve_num_bins(args.bin_size, args.bins)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.hist_1d and args.hist_rgb_1d:
        print("Error: use either --1d or --rgb-1d, not both.", file=sys.stderr)
        return 1

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Error: failed to read image: {args.image}", file=sys.stderr)
        return 1

    if args.emd is not None:
        if args.hist_1d or args.hist_rgb_1d:
            print("Error: --emd requires 3D color histograms (do not use --1d or --rgb-1d).", file=sys.stderr)
            return 1
        if not args.emd.is_file():
            print(f"Error: second image not found: {args.emd}", file=sys.stderr)
            return 1
        if args.emd_grid <= 0:
            print("Error: --emd-grid must be positive.", file=sys.stderr)
            return 1
        if args.emd_subcubes and args.emd_grid < 2:
            print("Error: --emd-grid must be at least 2 when using --emd-subcubes.", file=sys.stderr)
            return 1
        if args.emd_fps <= 0:
            print("Error: --emd-fps must be positive.", file=sys.stderr)
            return 1
        if args.emd_top_k <= 0:
            print("Error: --emd-top-k must be positive.", file=sys.stderr)
            return 1
        if args.emd_output is None and not args.show:
            print("Error: use --emd-output PATH and/or --show with --emd.", file=sys.stderr)
            return 1

        image_bgr_b = cv2.imread(str(args.emd), cv2.IMREAD_COLOR)
        if image_bgr_b is None:
            print(f"Error: failed to read image: {args.emd}", file=sys.stderr)
            return 1

        hist_a, bin_edges = compute_3d_histogram(image_bgr, num_bins, normalize=not args.no_normalize)
        hist_b, _bin_edges_b = compute_3d_histogram(image_bgr_b, num_bins, normalize=not args.no_normalize)
        if hist_a.shape != hist_b.shape:
            print("Error: histogram shapes differ; use the same --bins for both images.", file=sys.stderr)
            return 1

        print(f"Image A: {args.image.name} ({image_bgr.shape[1]}x{image_bgr.shape[0]})")
        print(f"Image B: {args.emd.name} ({image_bgr_b.shape[1]}x{image_bgr_b.shape[0]})")
        print(f"Bins per channel: {num_bins} ({num_bins}³ total bins)")
        if args.emd_subcubes:
            print(f"EMD mode: sub-cube grid ({args.emd_grid}×{args.emd_grid}×{args.emd_grid})")
        else:
            print("EMD mode: bin-level flows (no sub-cubes)")
        print(f"EMD top bins per image: {args.emd_top_k}")
        print(f"EMD ground metric: {args.emd_metric}")

        try:
            output_path = args.emd_output or Path(f"emd_{args.image.stem}_to_{args.emd.stem}.gif")
            visualize_emd_transport(
                args.image,
                args.emd,
                hist_a,
                hist_b,
                bin_edges,
                output_path=output_path,
                grid=args.emd_grid,
                fps=args.emd_fps,
                top_k=args.emd_top_k,
                metric=args.emd_metric,
                bins_only=not args.emd_subcubes,
                show=args.show,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.save_grayscale is not None:
        gray = save_grayscale_image(image_bgr, args.save_grayscale)
    elif args.hist_1d:
        gray = to_grayscale(image_bgr)
    else:
        gray = None

    if args.hist_1d:
        hist, edges = compute_1d_histogram(gray, num_bins, normalize=not args.no_normalize)
        title = f"1D Grayscale Histogram — {args.image.name} ({num_bins} bins)"
        print(f"Image: {args.image.name} ({image_bgr.shape[1]}x{image_bgr.shape[0]})")
        print(f"Bins: {num_bins} (bin size ≈ {256 / num_bins:.2f} intensity levels)")
        print(f"Non-zero bins: {int(np.count_nonzero(hist))}")
        print(f"Mode bin value: {hist.max():.6f}")
        print(f"Grayscale pixels: {gray.ravel().tolist()}")

        if args.save_data is not None:
            save_histogram_data(hist, args.save_data)

        wants_plot = args.output is not None or args.show or args.output_html is not None
        if wants_plot:
            if args.output is not None or args.show:
                plot_1d_histogram_matplotlib(
                    hist,
                    edges,
                    title=title,
                    output_path=args.output,
                    show=args.show,
                    normalized=not args.no_normalize,
                )
            if args.output_html is not None:
                try:
                    plot_1d_histogram_html(
                        hist,
                        edges,
                        title=title,
                        output_path=args.output_html,
                    )
                except RuntimeError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
        elif args.save_data is None and args.save_grayscale is None:
            print("No output requested. Use --output, --show, --output-html, or --save-data.")
        return 0

    if args.hist_rgb_1d:
        normalized = not args.no_normalize
        hists, edges = compute_rgb_1d_histograms(image_bgr, num_bins, normalize=normalized)
        print(f"Image: {args.image.name} ({image_bgr.shape[1]}x{image_bgr.shape[0]})")
        print(f"Bins per channel: {num_bins} (bin size ≈ {256 / num_bins:.2f} intensity levels)")
        for channel in ("Red", "Green", "Blue"):
            hist = hists[channel]
            print(
                f"{channel}: non-zero bins={int(np.count_nonzero(hist))}, "
                f"mode={hist.max():.6f}"
            )

        if args.save_data is not None:
            np.savez(
                args.save_data,
                red=hists["Red"],
                green=hists["Green"],
                blue=hists["Blue"],
                edges=edges,
            )
            print(f"Saved RGB histogram arrays to {args.save_data}")

        wants_plot = args.output is not None or args.show or args.output_html is not None
        if wants_plot:
            if args.output is not None or args.show:
                plot_rgb_1d_histograms_matplotlib(
                    hists,
                    edges,
                    image_name=args.image.name,
                    output_path=args.output,
                    show=args.show,
                    normalized=normalized,
                )
            if args.output_html is not None:
                try:
                    plot_rgb_1d_histograms_html(
                        hists,
                        edges,
                        image_name=args.image.name,
                        output_path=args.output_html,
                        normalized=normalized,
                    )
                except RuntimeError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
        elif args.save_data is None:
            print("No output requested. Use --output, --show, --output-html, or --save-data.")
        return 0

    hist, bin_edges = compute_3d_histogram(image_bgr, num_bins, normalize=not args.no_normalize)
    num_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    normalized = not args.no_normalize

    effective_bin_size = 256 / num_bins
    title = f"3D Color Histogram — {args.image.name} ({num_bins}³ bins)"
    print(f"Image: {args.image.name} ({image_bgr.shape[1]}x{image_bgr.shape[0]})")
    print(f"Bins per channel: {num_bins} (bin size ≈ {effective_bin_size:.2f} pixel values)")
    print(f"Histogram shape: {hist.shape} ({hist.size} total bins)")
    print(f"Non-zero bins: {int(np.count_nonzero(hist))}")
    print(f"Mode bin value: {hist.max():.6f}")

    if args.save_data is not None:
        save_histogram_data(hist, args.save_data)

    wants_plot = args.output is not None or args.show or args.output_html is not None
    if wants_plot:
        if args.output is not None or args.show:
            plot_3d_histogram_matplotlib(
                hist,
                bin_edges,
                title=title,
                output_path=args.output,
                show=args.show,
                min_fraction=args.min_fraction,
                normalized=normalized,
                num_pixels=num_pixels,
                show_counts=args.show_counts,
            )
        if args.output_html is not None:
            try:
                plot_3d_histogram_html(
                    hist,
                    bin_edges,
                    title=title,
                    output_path=args.output_html,
                    min_fraction=args.min_fraction,
                    normalized=normalized,
                    num_pixels=num_pixels,
                    show_counts=args.show_counts,
                )
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
    elif args.save_data is None:
        print("No output requested. Use --show, --output, --output-html, or --save-data.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
