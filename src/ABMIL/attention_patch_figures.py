"""Create held-out ABMIL attention-patch interpretation figures.

The checkpoint's ``test_slide_stems`` list is used by default so that figures
are generated only from slides excluded from model development.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image

from src.ABMIL.create_splits import COMPARISONS, load_label_file
from src.ABMIL.model import AttentionMIL
from src.config import DEVICE


PATCH_SIZE_UM = 500


def class_name(value: str | list[str]) -> str:
    return " or ".join(value) if isinstance(value, list) else value


def load_attention_model(checkpoint_path: Path) -> tuple[AttentionMIL, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = AttentionMIL(
        input_dim=1024, hidden_dim=256, attention_dim=128, dropout=0.5
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def attention_percentiles(attention: np.ndarray) -> np.ndarray:
    """Return percentile ranks in [0, 1], with the largest value equal to 1."""
    if attention.ndim != 1 or len(attention) == 0:
        raise ValueError("attention must be a non-empty one-dimensional array")
    order = np.argsort(attention, kind="stable")
    ranks = np.empty(len(attention), dtype=np.float64)
    ranks[order] = np.arange(len(attention))
    return ranks / max(len(attention) - 1, 1)


def select_attention_indices(attention: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Select top, central, and bottom ranked patches without overlap when possible."""
    if k < 1:
        raise ValueError("k must be positive")
    order = np.argsort(attention, kind="stable")
    effective_k = min(k, len(order))
    bottom = order[:effective_k]
    top = order[-effective_k:][::-1]
    start = max(0, (len(order) - effective_k) // 2)
    median = order[start : start + effective_k]
    return {"top": top, "median": median, "bottom": bottom}


@torch.no_grad()
def infer_slide(model: AttentionMIL, features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    logits, attention = model(features.to(DEVICE))
    probabilities = torch.softmax(logits, dim=0).cpu().numpy()
    return probabilities, attention.cpu().numpy()


def resolve_svs(svs_dir: Path, slide_id: str) -> Path:
    direct = svs_dir / f"{slide_id}.svs"
    if direct.exists():
        return direct
    matches = list(svs_dir.rglob(f"{slide_id}.svs"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one SVS for {slide_id} under {svs_dir}; found {len(matches)}"
        )
    return matches[0]


def read_patch(
    slide: openslide.OpenSlide, coordinate: np.ndarray, mpp: float, display_size: int
) -> Image.Image:
    patch_size = max(1, int(round(PATCH_SIZE_UM / mpp)))
    patch = slide.read_region(
        (int(coordinate[0]), int(coordinate[1])),
        level=0,
        size=(patch_size, patch_size),
    ).convert("RGB")
    return patch.resize((display_size, display_size), Image.Resampling.LANCZOS)


def attention_heatmap_arrays(
    slide: openslide.OpenSlide,
    coords: np.ndarray,
    percentiles: np.ndarray,
    mpp: float,
    thumb_scale: int,
) -> tuple[np.ndarray, np.ma.MaskedArray]:
    width, height = slide.dimensions
    thumbnail = np.asarray(
        slide.get_thumbnail(
            (max(1, width // thumb_scale), max(1, height // thumb_scale))
        ).convert("RGB")
    )
    thumb_height, thumb_width = thumbnail.shape[:2]
    tile_size = max(1, int(round(PATCH_SIZE_UM / mpp)) // thumb_scale)
    heat = np.full((thumb_height, thumb_width), np.nan, dtype=np.float32)

    for (x, y), value in zip(coords, percentiles):
        x0, y0 = int(x / thumb_scale), int(y / thumb_scale)
        x1, y1 = min(x0 + tile_size, thumb_width), min(y0 + tile_size, thumb_height)
        if x1 <= x0 or y1 <= y0:
            continue
        region = heat[y0:y1, x0:x1]
        heat[y0:y1, x0:x1] = np.where(
            np.isnan(region) | (value > region), value, region
        )
    return thumbnail, np.ma.masked_invalid(heat)


def collect_heldout_attention(
    model: AttentionMIL,
    checkpoint: dict[str, Any],
    features_dir: Path,
    slide_labels: dict[str, tuple[int, Any]],
    comparison: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    checkpoint_comparison = checkpoint.get("comparison")
    if checkpoint_comparison != comparison:
        raise ValueError(
            f"Checkpoint comparison is {checkpoint_comparison}, requested {comparison}"
        )
    test_stems = checkpoint.get("test_slide_stems")
    if not test_stems:
        raise ValueError("Checkpoint has no held-out test_slide_stems")

    rows: list[dict[str, Any]] = []
    slides: dict[str, dict[str, Any]] = {}
    for position, slide_id in enumerate(test_stems, start=1):
        feature_path = features_dir / f"{slide_id}.pt"
        if not feature_path.exists() or slide_id not in slide_labels:
            raise FileNotFoundError(f"Missing held-out features or label for {slide_id}")
        data = torch.load(feature_path, map_location="cpu", weights_only=False)
        features = data["features"]
        coords_raw = data["coords"]
        coords = coords_raw.numpy() if torch.is_tensor(coords_raw) else np.asarray(coords_raw)
        if len(features) != len(coords):
            raise ValueError(f"Feature/coordinate mismatch for {slide_id}")

        probabilities, attention = infer_slide(model, features)
        if len(attention) != len(coords):
            raise ValueError(f"Attention/coordinate mismatch for {slide_id}")
        percentiles = attention_percentiles(attention)
        true_label, case_id = slide_labels[slide_id]
        prediction = int(np.argmax(probabilities))
        selection = select_attention_indices(attention, k=min(5, len(attention)))
        group_by_index = {
            int(index): group for group, indices in selection.items() for index in indices
        }

        slides[slide_id] = {
            "slide_id": slide_id,
            "case_id": str(case_id),
            "true_label": int(true_label),
            "prediction": prediction,
            "correct": prediction == int(true_label),
            "confidence": float(probabilities[prediction]),
            "probabilities": probabilities,
            "attention": attention,
            "percentiles": percentiles,
            "coords": coords,
            "mpp": float(data["mpp"]),
            "feature_path": feature_path,
        }
        ranks_desc = np.empty(len(attention), dtype=int)
        ranks_desc[np.argsort(attention)[::-1]] = np.arange(1, len(attention) + 1)
        for index, ((x, y), raw, percentile, rank) in enumerate(
            zip(coords, attention, percentiles, ranks_desc)
        ):
            rows.append(
                {
                    "slide_id": slide_id,
                    "case_id": str(case_id),
                    "true_label": int(true_label),
                    "predicted_label": prediction,
                    "correct": prediction == int(true_label),
                    "probability_class_0": float(probabilities[0]),
                    "probability_class_1": float(probabilities[1]),
                    "patch_index": index,
                    "x": int(x),
                    "y": int(y),
                    "raw_attention": float(raw),
                    "attention_rank": int(rank),
                    "attention_percentile": float(percentile),
                    "selection_group": group_by_index.get(index, "other"),
                }
            )
        print(
            f"[{position}/{len(test_stems)}] {slide_id}: true={true_label}, "
            f"pred={prediction}, confidence={probabilities[prediction]:.3f}"
        )
    return pd.DataFrame(rows), slides


def choose_representative_slides(
    slides: dict[str, dict[str, Any]], per_class: int
) -> dict[int, list[dict[str, Any]]]:
    selected: dict[int, list[dict[str, Any]]] = {}
    for label in [0, 1]:
        correct = [
            slide
            for slide in slides.values()
            if slide["true_label"] == label and slide["correct"]
        ]
        correct.sort(key=lambda slide: (-slide["confidence"], slide["slide_id"]))
        if not correct:
            raise ValueError(f"No correctly predicted held-out slides for class {label}")
        selected[label] = correct[:per_class]
    return selected


def add_patch_axis(
    ax: plt.Axes,
    patch: Image.Image,
    rank: int,
    percentile: float,
) -> None:
    ax.imshow(patch)
    ax.set_title(f"Rank {rank}\n{100 * percentile:.1f}th pct", fontsize=9)
    ax.axis("off")


def create_highest_attention_figure(
    selected: dict[int, list[dict[str, Any]]],
    svs_dir: Path,
    names: list[str],
    top_k: int,
    display_size: int,
    output_path: Path,
) -> None:
    rows = [(label, slide) for label in [0, 1] for slide in selected[label]]
    fig, axes = plt.subplots(len(rows), top_k, figsize=(2.6 * top_k, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for row, (label, record) in enumerate(rows):
        indices = select_attention_indices(record["attention"], top_k)["top"]
        slide = openslide.OpenSlide(str(resolve_svs(svs_dir, record["slide_id"])))
        for column, index in enumerate(indices):
            patch = read_patch(slide, record["coords"][index], record["mpp"], display_size)
            add_patch_axis(
                axes[row, column],
                patch,
                column + 1,
                record["percentiles"][index],
            )
        slide.close()
        axes[row, 0].text(
            -0.08,
            0.5,
            f"{names[label]}\n{record['slide_id']}\n"
            f"Pred: {names[record['prediction']]} ({record['confidence']:.2f})",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
    fig.suptitle("Highest-attention vessel patches by diagnosis", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_attention_level_figure(
    selected: dict[int, list[dict[str, Any]]],
    svs_dir: Path,
    names: list[str],
    top_k: int,
    display_size: int,
    output_path: Path,
) -> None:
    rows = [(label, group) for label in [0, 1] for group in ["top", "median", "bottom"]]
    fig, axes = plt.subplots(len(rows), top_k, figsize=(2.6 * top_k, 2.7 * len(rows)))
    for row, (label, group) in enumerate(rows):
        record = selected[label][0]
        indices = select_attention_indices(record["attention"], top_k)[group]
        slide = openslide.OpenSlide(str(resolve_svs(svs_dir, record["slide_id"])))
        for column, index in enumerate(indices):
            patch = read_patch(slide, record["coords"][index], record["mpp"], display_size)
            add_patch_axis(
                axes[row, column],
                patch,
                int(np.sum(record["attention"] > record["attention"][index])) + 1,
                record["percentiles"][index],
            )
        slide.close()
        axes[row, 0].text(
            -0.08,
            0.5,
            f"{names[label]} — {group.title()}\n{record['slide_id']}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
    fig.suptitle("Highest, median, and lowest-attention vessel patches", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def create_spatial_context_figure(
    selected: dict[int, list[dict[str, Any]]],
    svs_dir: Path,
    names: list[str],
    top_k: int,
    display_size: int,
    thumb_scale: int,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 7))
    grid = fig.add_gridspec(2, top_k + 2, width_ratios=[2.2, 2.2] + [1] * top_k)
    heat_image = None
    for row, label in enumerate([0, 1]):
        record = selected[label][0]
        slide = openslide.OpenSlide(str(resolve_svs(svs_dir, record["slide_id"])))
        thumbnail, heat = attention_heatmap_arrays(
            slide,
            record["coords"],
            record["percentiles"],
            record["mpp"],
            thumb_scale,
        )
        ax_thumb = fig.add_subplot(grid[row, 0])
        ax_thumb.imshow(thumbnail)
        ax_thumb.set_title(
            f"{names[label]}: {record['slide_id']}\n"
            f"Pred {names[record['prediction']]} ({record['confidence']:.2f})"
        )
        ax_thumb.axis("off")

        ax_heat = fig.add_subplot(grid[row, 1])
        ax_heat.imshow(thumbnail)
        heat_image = ax_heat.imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax_heat.set_title("Attention percentile")
        ax_heat.axis("off")

        indices = select_attention_indices(record["attention"], top_k)["top"]
        for column, index in enumerate(indices, start=2):
            ax = fig.add_subplot(grid[row, column])
            patch = read_patch(slide, record["coords"][index], record["mpp"], display_size)
            add_patch_axis(
                ax, patch, column - 1, record["percentiles"][index]
            )
        slide.close()
    if heat_image is not None:
        colorbar_axis = fig.add_axes((0.955, 0.2, 0.012, 0.6))
        fig.colorbar(heat_image, cax=colorbar_axis, label="Attention percentile")
    fig.suptitle("Whole-slide attention context and highest-attention patches", fontsize=17)
    fig.subplots_adjust(left=0.02, right=0.94, top=0.88, bottom=0.03, wspace=0.08)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create held-out MIL attention figures")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--svs-dir", type=Path, required=True)
    parser.add_argument("--labels-xlsx", type=Path, default=Path("data/case_labels.xlsx"))
    parser.add_argument("--comparison", choices=sorted(COMPARISONS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--slides-per-class", type=int, default=2)
    parser.add_argument("--display-size", type=int, default=224)
    parser.add_argument("--thumb-scale", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_attention_model(args.checkpoint)
    slide_labels, comparison = load_label_file(args.labels_xlsx, args.comparison)
    names = [class_name(comparison["class_0"]), class_name(comparison["class_1"])]

    attention_table, slides = collect_heldout_attention(
        model, checkpoint, args.features_dir, slide_labels, args.comparison
    )
    attention_table["true_diagnosis"] = attention_table["true_label"].map(
        dict(enumerate(names))
    )
    attention_table["predicted_diagnosis"] = attention_table["predicted_label"].map(
        dict(enumerate(names))
    )
    attention_table.to_csv(args.output_dir / "attention_results.csv", index=False)

    selected = choose_representative_slides(slides, args.slides_per_class)
    selected_rows = []
    for label, records in selected.items():
        for record in records:
            selected_rows.append(
                {
                    "slide_id": record["slide_id"],
                    "true_diagnosis": names[label],
                    "predicted_diagnosis": names[record["prediction"]],
                    "confidence": record["confidence"],
                    "selection_rule": "highest-confidence correct held-out slide",
                }
            )
    pd.DataFrame(selected_rows).to_csv(
        args.output_dir / "selected_slides.csv", index=False
    )

    create_highest_attention_figure(
        selected,
        args.svs_dir,
        names,
        args.top_k,
        args.display_size,
        args.output_dir / "figure_1_highest_attention_by_diagnosis.png",
    )
    create_attention_level_figure(
        selected,
        args.svs_dir,
        names,
        args.top_k,
        args.display_size,
        args.output_dir / "figure_2_top_median_bottom.png",
    )
    create_spatial_context_figure(
        selected,
        args.svs_dir,
        names,
        args.top_k,
        args.display_size,
        args.thumb_scale,
        args.output_dir / "figure_3_whole_slide_context.png",
    )

    metadata = {
        "comparison": args.comparison,
        "checkpoint": str(args.checkpoint),
        "features_dir": str(args.features_dir),
        "split": "checkpoint held-out test set",
        "n_test_slides": len(slides),
        "checkpoint_test_auc": checkpoint.get("test_auc"),
        "top_k": args.top_k,
        "slides_per_class": args.slides_per_class,
        "selected_slides": selected_rows,
        "attention_interpretation": (
            "Attention is relative within each slide and is not class-specific evidence."
        ),
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved attention figures to {args.output_dir}")


if __name__ == "__main__":
    main()
