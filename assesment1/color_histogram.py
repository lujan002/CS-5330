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
