"""PCA and UMAP analysis of DINOv2 vessel embeddings.

The feature extractor writes one ``.pt`` file per slide under ``all/``,
``vessels_g/``, and ``vessels_w/``.  This module joins those files to the
existing clinical label spreadsheet and creates comparable patch- and
slide-level visualizations for every vessel subset.

Example
-------
python -m src.visualization.embedding_analysis \
    --features-root data/processed_vessels \
    --labels-xlsx /path/to/labels.xlsx \
    --comparison control_vs_CTE \
    --output-dir outputs/embedding_analysis
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.ABMIL.create_splits import COMPARISONS, load_label_file


VESSEL_SUBSETS = {
    "all_vessels": "all",
    "grey_vessels": "vessels_g",
    "white_vessels": "vessels_w",
}


@dataclass
class EmbeddingSet:
    features: np.ndarray
    labels: np.ndarray
    slide_ids: np.ndarray
    case_ids: np.ndarray

    def subset(self, indices: np.ndarray) -> "EmbeddingSet":
        return EmbeddingSet(
            features=self.features[indices],
            labels=self.labels[indices],
            slide_ids=self.slide_ids[indices],
            case_ids=self.case_ids[indices],
        )


def _class_display_name(value: str | list[str]) -> str:
    return " + ".join(value) if isinstance(value, list) else str(value)


def class_names_for_comparison(comparison: dict[str, Any]) -> dict[int, str]:
    return {
        0: _class_display_name(comparison["class_0"]),
        1: _class_display_name(comparison["class_1"]),
    }


def _load_feature_tensor(pt_path: Path) -> np.ndarray:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "features" not in data:
        raise KeyError(f"{pt_path} does not contain a 'features' tensor")

    features = data["features"]
    if isinstance(features, torch.Tensor):
        features = features.detach().float().numpy()
    else:
        features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            f"Expected a non-empty 2D feature matrix in {pt_path}; "
            f"received shape {features.shape}"
        )
    return features


def load_patch_embeddings(
    feature_dir: Path,
    slide_labels: dict[str, tuple[int, Any]],
    patches_per_slide: int,
    seed: int,
) -> EmbeddingSet:
    """Load at most ``patches_per_slide`` embeddings from each labeled slide."""
    rng = np.random.default_rng(seed)
    feature_parts: list[np.ndarray] = []
    labels: list[int] = []
    slide_ids: list[str] = []
    case_ids: list[str] = []

    for pt_path in sorted(feature_dir.glob("*.pt")):
        if pt_path.stem not in slide_labels:
            continue
        label, case_id = slide_labels[pt_path.stem]
        features = _load_feature_tensor(pt_path)
        n_select = min(patches_per_slide, len(features))
        indices = rng.choice(len(features), size=n_select, replace=False)

        feature_parts.append(features[indices])
        labels.extend([int(label)] * n_select)
        slide_ids.extend([pt_path.stem] * n_select)
        case_ids.extend([str(case_id)] * n_select)

    if not feature_parts:
        raise RuntimeError(f"No labeled .pt feature files found in {feature_dir}")

    return EmbeddingSet(
        features=np.concatenate(feature_parts, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        slide_ids=np.asarray(slide_ids),
        case_ids=np.asarray(case_ids),
    )


def load_slide_embeddings(
    feature_dir: Path,
    slide_labels: dict[str, tuple[int, Any]],
) -> EmbeddingSet:
    """Create one slide vector by averaging all of its patch embeddings."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    slide_ids: list[str] = []
    case_ids: list[str] = []

    for pt_path in sorted(feature_dir.glob("*.pt")):
        if pt_path.stem not in slide_labels:
            continue
        label, case_id = slide_labels[pt_path.stem]
        patch_features = _load_feature_tensor(pt_path)
        features.append(patch_features.mean(axis=0))
        labels.append(int(label))
        slide_ids.append(pt_path.stem)
        case_ids.append(str(case_id))

    if not features:
        raise RuntimeError(f"No labeled .pt feature files found in {feature_dir}")

    return EmbeddingSet(
        features=np.stack(features),
        labels=np.asarray(labels, dtype=np.int64),
        slide_ids=np.asarray(slide_ids),
        case_ids=np.asarray(case_ids),
    )


def balance_patch_classes(data: EmbeddingSet, seed: int) -> EmbeddingSet:
    """Downsample patch classes to equal size after per-slide sampling."""
    classes = np.unique(data.labels)
    if len(classes) != 2:
        raise ValueError(f"Expected two classes, found {classes.tolist()}")

    rng = np.random.default_rng(seed)
    per_class = [np.flatnonzero(data.labels == label) for label in classes]
    target = min(len(indices) for indices in per_class)
    selected = np.concatenate(
        [rng.choice(indices, size=target, replace=False) for indices in per_class]
    )
    rng.shuffle(selected)
    return data.subset(selected)


def reduce_embeddings(
    data: EmbeddingSet,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize, calculate 2D PCA, and run UMAP on up to 50 PCs."""
    n_samples, n_features = data.features.shape
    if n_samples < 3:
        raise ValueError("At least three samples are required for PCA/UMAP")
    if len(np.unique(data.labels)) != 2:
        raise ValueError("Both diagnostic classes must be present")

    scaled = StandardScaler().fit_transform(data.features)
    pca_2d = PCA(n_components=2, random_state=seed)
    pca_coords = pca_2d.fit_transform(scaled)

    pre_components = min(50, n_samples - 1, n_features)
    pca_pre = PCA(n_components=pre_components, random_state=seed)
    pca_features = pca_pre.fit_transform(scaled)

    effective_neighbors = min(n_neighbors, n_samples - 1)
    if effective_neighbors < 2:
        raise ValueError("UMAP requires at least two neighbors")
    umap_coords = UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(pca_features)

    table = pd.DataFrame(
        {
            "pca_1": pca_coords[:, 0],
            "pca_2": pca_coords[:, 1],
            "umap_1": umap_coords[:, 0],
            "umap_2": umap_coords[:, 1],
            "label": data.labels,
            "slide_id": data.slide_ids,
            "case_id": data.case_ids,
        }
    )

    silhouette_samples = min(10_000, n_samples)
    metrics = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "n_slides": int(len(np.unique(data.slide_ids))),
        "n_cases": int(len(np.unique(data.case_ids))),
        "class_counts": {
            str(label): int(np.sum(data.labels == label))
            for label in np.unique(data.labels)
        },
        "pca_2d_explained_variance": pca_2d.explained_variance_ratio_.tolist(),
        "pca_preprocessing_components": int(pre_components),
        "pca_preprocessing_explained_variance": float(
            pca_pre.explained_variance_ratio_.sum()
        ),
        "umap_effective_neighbors": int(effective_neighbors),
        "silhouette_sample_size": int(silhouette_samples),
        "silhouette_pca_space_cosine": float(
            silhouette_score(
                pca_features,
                data.labels,
                metric="cosine",
                sample_size=silhouette_samples,
                random_state=seed,
            )
        ),
    }
    return table, metrics


def plot_embedding(
    table: pd.DataFrame,
    method: str,
    title: str,
    output_path: Path,
    point_size: float,
    alpha: float,
    pca_variance: list[float] | None = None,
) -> None:
    x, y = f"{method}_1", f"{method}_2"
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=table,
        x=x,
        y=y,
        hue="class_name",
        hue_order=sorted(table["class_name"].unique()),
        alpha=alpha,
        s=point_size,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(title)
    if method == "pca" and pca_variance is not None:
        ax.set_xlabel(f"PC1 ({100 * pca_variance[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({100 * pca_variance[1]:.1f}% variance)")
    else:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    ax.legend(title="Diagnosis", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_level(
    data: EmbeddingSet,
    level: str,
    experiment_name: str,
    class_names: dict[int, str],
    output_dir: Path,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    table, metrics = reduce_embeddings(
        data, seed=seed, n_neighbors=n_neighbors, min_dist=min_dist
    )
    table["class_name"] = table["label"].map(class_names)
    table.to_csv(output_dir / f"{level}_coordinates.csv", index=False)

    title_prefix = experiment_name.replace("_", " ").title()
    size, alpha = (12, 0.35) if level == "patch" else (75, 0.85)
    plot_embedding(
        table,
        method="pca",
        title=f"{title_prefix}: {level.title()}-level PCA",
        output_path=output_dir / f"{level}_pca.png",
        point_size=size,
        alpha=alpha,
        pca_variance=metrics["pca_2d_explained_variance"],
    )
    plot_embedding(
        table,
        method="umap",
        title=f"{title_prefix}: {level.title()}-level UMAP",
        output_path=output_dir / f"{level}_umap.png",
        point_size=size,
        alpha=alpha,
    )
    return table, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA and UMAP plots for vessel DINOv2 embeddings"
    )
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--labels-xlsx", type=Path, required=True)
    parser.add_argument("--comparison", choices=sorted(COMPARISONS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--patches-per-slide", type=int, default=500)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patches_per_slide < 1:
        raise ValueError("--patches-per-slide must be positive")
    if args.n_neighbors < 2:
        raise ValueError("--n-neighbors must be at least 2")
    if not 0 <= args.min_dist <= 1:
        raise ValueError("--min-dist must be between 0 and 1")

    slide_labels, comparison = load_label_file(
        str(args.labels_xlsx), args.comparison, sheet_name=args.sheet_name
    )
    class_names = class_names_for_comparison(comparison)
    run_root = args.output_dir / args.comparison
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "comparison": args.comparison,
        "comparison_name": comparison["name"],
        "class_names": class_names,
        "seed": args.seed,
        "patches_per_slide": args.patches_per_slide,
        "umap": {
            "requested_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
        },
        "experiments": {},
    }

    sns.set_theme(style="whitegrid", context="notebook")
    for experiment_name, folder_name in VESSEL_SUBSETS.items():
        feature_dir = args.features_root / folder_name
        if not feature_dir.is_dir():
            raise FileNotFoundError(f"Missing vessel feature directory: {feature_dir}")

        experiment_dir = run_root / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{experiment_name}] Loading {feature_dir}")

        patch_data = load_patch_embeddings(
            feature_dir,
            slide_labels,
            patches_per_slide=args.patches_per_slide,
            seed=args.seed,
        )
        patch_data = balance_patch_classes(patch_data, seed=args.seed)
        _, patch_metrics = run_level(
            patch_data,
            "patch",
            experiment_name,
            class_names,
            experiment_dir,
            args.seed,
            args.n_neighbors,
            args.min_dist,
        )

        slide_data = load_slide_embeddings(feature_dir, slide_labels)
        _, slide_metrics = run_level(
            slide_data,
            "slide",
            experiment_name,
            class_names,
            experiment_dir,
            args.seed,
            args.n_neighbors,
            args.min_dist,
        )
        summary["experiments"][experiment_name] = {
            "feature_dir": str(feature_dir),
            "patch": patch_metrics,
            "slide": slide_metrics,
        }
        print(
            f"[{experiment_name}] {patch_metrics['n_samples']} balanced patches, "
            f"{slide_metrics['n_samples']} slides"
        )

    with (run_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved embedding analysis to {run_root}")


if __name__ == "__main__":
    main()
