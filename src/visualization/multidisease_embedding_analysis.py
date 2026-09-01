"""Visualize vessel-patch embeddings across all four disease groups.

This is the multi-class companion to ``embedding_analysis.py``. It produces
PCA and UMAP views for all vessels, grey-matter vessels, and white-matter
vessels, with one point per patch and a consistent disease color palette.

Example
-------
python -m src.visualization.multidisease_embedding_analysis \
    --features-root data/processed_vessels \
    --labels-xlsx data/case_labels.xlsx \
    --output-dir visualizations/embedding_analysis/by_disease
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
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.visualization.embedding_analysis import (
    EmbeddingSet,
    VESSEL_SUBSETS,
    load_patch_embeddings,
)


DISEASE_ORDER = ["Control", "RHI", "Low CTE", "High CTE"]
DISEASE_TO_LABEL = {name: index for index, name in enumerate(DISEASE_ORDER)}
DISEASE_COLORS = {
    "Control": "#4C72B0",
    "RHI": "#DD8452",
    "Low CTE": "#55A868",
    "High CTE": "#C44E52",
}


def load_disease_labels(
    label_file: Path, sheet_name: int | str = 0
) -> dict[str, tuple[int, Any]]:
    """Return ``slide stem -> (four-class label, case ID)`` from the workbook."""
    frame = pd.read_excel(label_file, sheet_name=sheet_name, engine="openpyxl")
    if len(frame.columns) < 9:
        raise ValueError("The label workbook must contain at least nine columns")
    frame.columns = [
        "case_id",
        "age",
        "block_id",
        "stain",
        "description",
        "scanner",
        "mag",
        "filename",
        "path_group",
        *[f"extra_{i}" for i in range(len(frame.columns) - 9)],
    ]
    frame = frame[frame["path_group"].isin(DISEASE_ORDER)].copy()
    frame["label"] = frame["path_group"].map(DISEASE_TO_LABEL)
    frame["slide_id"] = frame["filename"].astype(str).str.replace(
        ".svs", "", regex=False
    )
    if frame["slide_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["slide_id"].duplicated(), "slide_id"])
        raise ValueError(f"Duplicate slide IDs in label workbook: {duplicates[:5]}")
    return {
        row.slide_id: (int(row.label), row.case_id)
        for row in frame.itertuples(index=False)
    }


def balance_diseases(data: EmbeddingSet, seed: int) -> EmbeddingSet:
    """Downsample every disease to the size of the smallest disease group."""
    observed = set(np.unique(data.labels).tolist())
    expected = set(range(len(DISEASE_ORDER)))
    if observed != expected:
        missing = [DISEASE_ORDER[label] for label in sorted(expected - observed)]
        raise ValueError(f"Missing disease groups: {missing}")

    rng = np.random.default_rng(seed)
    indices_by_disease = [
        np.flatnonzero(data.labels == label) for label in range(len(DISEASE_ORDER))
    ]
    target = min(len(indices) for indices in indices_by_disease)
    selected = np.concatenate(
        [rng.choice(indices, size=target, replace=False) for indices in indices_by_disease]
    )
    rng.shuffle(selected)
    return data.subset(selected)


def reduce_multidisease_embeddings(
    data: EmbeddingSet,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create PCA/UMAP coordinates and unsupervised separation diagnostics."""
    n_samples, n_features = data.features.shape
    scaled = StandardScaler().fit_transform(data.features)

    pca_2d = PCA(n_components=2, random_state=seed)
    pca_coordinates = pca_2d.fit_transform(scaled)

    preprocessing_components = min(50, n_samples - 1, n_features)
    pca_preprocessor = PCA(
        n_components=preprocessing_components, random_state=seed
    )
    pca_features = pca_preprocessor.fit_transform(scaled)

    effective_neighbors = min(n_neighbors, n_samples - 1)
    umap_coordinates = UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(pca_features)

    table = pd.DataFrame(
        {
            "pca_1": pca_coordinates[:, 0],
            "pca_2": pca_coordinates[:, 1],
            "umap_1": umap_coordinates[:, 0],
            "umap_2": umap_coordinates[:, 1],
            "label": data.labels,
            "disease": [DISEASE_ORDER[label] for label in data.labels],
            "slide_id": data.slide_ids,
            "case_id": data.case_ids,
        }
    )

    silhouette_samples = min(10_000, n_samples)
    metrics = {
        "n_patches": int(n_samples),
        "n_features": int(n_features),
        "n_slides": int(len(np.unique(data.slide_ids))),
        "n_cases": int(len(np.unique(data.case_ids))),
        "disease_counts": {
            disease: int(np.sum(table["disease"] == disease))
            for disease in DISEASE_ORDER
        },
        "pca_2d_explained_variance": pca_2d.explained_variance_ratio_.tolist(),
        "pca_preprocessing_components": int(preprocessing_components),
        "pca_preprocessing_explained_variance": float(
            pca_preprocessor.explained_variance_ratio_.sum()
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


def draw_embedding(
    ax: plt.Axes,
    table: pd.DataFrame,
    method: str,
    title: str,
    pca_variance: list[float] | None = None,
    show_legend: bool = True,
) -> None:
    sns.scatterplot(
        data=table,
        x=f"{method}_1",
        y=f"{method}_2",
        hue="disease",
        hue_order=DISEASE_ORDER,
        palette=DISEASE_COLORS,
        alpha=0.35,
        s=11,
        linewidth=0,
        ax=ax,
        legend=show_legend,
    )
    ax.set_title(title)
    if method == "pca" and pca_variance is not None:
        ax.set_xlabel(f"PC1 ({100 * pca_variance[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({100 * pca_variance[1]:.1f}% variance)")
    else:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    if show_legend:
        ax.legend(title="Disease", markerscale=2, frameon=True)


def save_individual_plots(
    table: pd.DataFrame,
    metrics: dict[str, Any],
    subset_title: str,
    output_dir: Path,
) -> None:
    for method in ["pca", "umap"]:
        fig, ax = plt.subplots(figsize=(9, 7))
        draw_embedding(
            ax,
            table,
            method,
            f"{subset_title}: Vessel-patch {method.upper()}",
            pca_variance=metrics["pca_2d_explained_variance"],
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"patch_{method}.png", dpi=300)
        plt.close(fig)


def save_combined_figure(
    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]], output_path: Path
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    titles = {
        "all_vessels": "All vessels",
        "grey_vessels": "Grey-matter vessels",
        "white_vessels": "White-matter vessels",
    }
    for row, subset in enumerate(VESSEL_SUBSETS):
        table, metrics = results[subset]
        draw_embedding(
            axes[row, 0],
            table,
            "pca",
            f"{titles[subset]} — PCA",
            pca_variance=metrics["pca_2d_explained_variance"],
            show_legend=False,
        )
        draw_embedding(
            axes[row, 1],
            table,
            "umap",
            f"{titles[subset]} — UMAP",
            show_legend=(row == 0),
        )
    fig.suptitle("Visualization of vessel-patch embeddings by disease", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize vessel-patch embeddings across four disease groups"
    )
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--labels-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--patches-per-slide", type=int, default=250)
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

    slide_labels = load_disease_labels(args.labels_xlsx, args.sheet_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    results: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "title": "Visualization of vessel-patch embeddings by disease",
        "disease_order": DISEASE_ORDER,
        "disease_colors": DISEASE_COLORS,
        "seed": args.seed,
        "patches_per_slide": args.patches_per_slide,
        "umap": {
            "requested_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
        },
        "experiments": {},
    }

    for subset_name, folder_name in VESSEL_SUBSETS.items():
        feature_dir = args.features_root / folder_name
        if not feature_dir.is_dir():
            raise FileNotFoundError(f"Missing vessel feature directory: {feature_dir}")
        subset_dir = args.output_dir / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{subset_name}] Loading {feature_dir}")
        embeddings = load_patch_embeddings(
            feature_dir,
            slide_labels,
            patches_per_slide=args.patches_per_slide,
            seed=args.seed,
        )
        embeddings = balance_diseases(embeddings, seed=args.seed)
        table, metrics = reduce_multidisease_embeddings(
            embeddings,
            seed=args.seed,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
        table.to_csv(subset_dir / "patch_coordinates.csv", index=False)
        save_individual_plots(
            table, metrics, subset_name.replace("_", " ").title(), subset_dir
        )
        results[subset_name] = (table, metrics)
        summary["experiments"][subset_name] = {
            "feature_dir": str(feature_dir),
            **metrics,
        }
        print(
            f"[{subset_name}] {metrics['n_patches']} balanced patches from "
            f"{metrics['n_slides']} slides"
        )

    save_combined_figure(
        results, args.output_dir / "vessel_patch_embeddings_by_disease.png"
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved multi-disease analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
