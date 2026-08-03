# -*- coding: utf-8 -*-
"""
run_clustering_evaluation.py

Reproducible clustering and evaluation pipeline for document embeddings.

Implemented algorithms
----------------------
1. K-Means++.
2. Spherical K-Means.
3. Hierarchical Agglomerative Clustering (HAC).
4. Spectral Co-Clustering.

Evaluation metrics
------------------
- Accuracy using optimal one-to-one Hungarian assignment.
- Macro F1-score.
- Weighted F1-score.
- Purity.
- Normalized Mutual Information (NMI).
- Adjusted Rand Index (ARI).
- V-measure.
- Normalized entropy.

Reproducibility
---------------
- No hard-coded local paths.
- Configurable command-line arguments.
- Explicit random seeds.
- HAC is executed once because it is deterministic.
- Stochastic algorithms are executed for all supplied seeds.
- Spectral Co-Clustering receives feature-wise Min-Max-scaled
  non-negative input, as required by the algorithm.
- Existing outputs may be skipped unless --overwrite is supplied.

Expected embedding-file columns
-------------------------------
Metadata:
    text or document
    ProcessedDocument or processed_document
    id
    label

Embedding features:
    emb_0, emb_1, ..., emb_n

Example
-------
python run_clustering_evaluation.py ^
    --input_dir outputs/embeddings ^
    --output_dir outputs/clustering ^
    --pattern "*.csv" ^
    --algorithms kmeansplusplus spherical_kmeans hac spectral_coclustering ^
    --seeds 1 2 3 4 5 6 7 8 9 10 ^
    --n_init 10 ^
    --max_iter 100
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralCoclustering,
)
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize


LOGGER = logging.getLogger("clustering_evaluation")

SUPPORTED_ALGORITHMS = (
    "kmeansplusplus",
    "spherical_kmeans",
    "hac",
    "spectral_coclustering",
)

METRIC_COLUMNS = (
    "F1_macro",
    "F1_weighted",
    "Purity",
    "NMI",
    "ARI",
    "Entropy",
    "Accuracy",
    "V_measure",
)


@dataclass(frozen=True)
class ClusteringSettings:
    """Configuration shared by all clustering algorithms."""

    algorithms: Tuple[str, ...]
    seeds: Tuple[int, ...]
    n_init: int = 10
    max_iter: int = 100
    hac_linkage: str = "ward"
    save_assignments: bool = True
    overwrite: bool = False

    def validate(self) -> None:
        invalid = [
            algorithm
            for algorithm in self.algorithms
            if algorithm not in SUPPORTED_ALGORITHMS
        ]
        if invalid:
            raise ValueError(
                f"Unsupported algorithms: {invalid}. "
                f"Supported values: {list(SUPPORTED_ALGORITHMS)}"
            )

        if not self.seeds:
            raise ValueError("At least one random seed is required.")

        if self.n_init <= 0:
            raise ValueError("n_init must be greater than zero.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than zero.")


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def set_global_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def find_first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> Optional[str]:
    """Return the first matching column name."""
    for column in candidates:
        if column in frame.columns:
            return column

    if required:
        raise ValueError(
            f"None of the required columns were found: {list(candidates)}. "
            f"Available columns: {list(frame.columns)}"
        )

    return None


def validate_embedding_file(frame: pd.DataFrame) -> List[str]:
    """Validate metadata and embedding columns."""
    if "label" not in frame.columns:
        raise ValueError("Input file does not contain a 'label' column.")

    if "id" not in frame.columns:
        raise ValueError("Input file does not contain an 'id' column.")

    embedding_columns = [
        column
        for column in frame.columns
        if column.startswith("emb_")
    ]

    if not embedding_columns:
        raise ValueError(
            "No embedding columns were found. Expected columns named "
            "emb_0, emb_1, ..."
        )

    return embedding_columns


def build_contingency_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a predicted-cluster by reference-class contingency matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    predicted_labels = np.unique(y_pred)

    matrix = np.zeros(
        (len(predicted_labels), len(true_labels)),
        dtype=np.int64,
    )

    for row_index, predicted_label in enumerate(predicted_labels):
        for column_index, true_label in enumerate(true_labels):
            matrix[row_index, column_index] = np.sum(
                (y_pred == predicted_label)
                & (y_true == true_label)
            )

    return matrix, predicted_labels, true_labels


def map_clusters_to_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Map clusters to classes using optimal Hungarian assignment."""
    matrix, predicted_labels, true_labels = build_contingency_matrix(
        y_true,
        y_pred,
    )

    row_indices, column_indices = linear_sum_assignment(-matrix)

    mapping = {
        predicted_labels[row]: true_labels[column]
        for row, column in zip(row_indices, column_indices)
    }

    # This fallback protects against unmatched cluster labels in rectangular
    # contingency matrices, although the experiments use K equal to class count.
    default_label = true_labels[0]

    return np.asarray(
        [
            mapping.get(label, default_label)
            for label in y_pred
        ]
    )


def clustering_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute clustering accuracy using Hungarian one-to-one assignment."""
    matrix, _, _ = build_contingency_matrix(y_true, y_pred)
    row_indices, column_indices = linear_sum_assignment(-matrix)

    return float(
        matrix[row_indices, column_indices].sum()
        / len(y_true)
    )


def purity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute majority-based cluster purity."""
    total = 0

    for cluster in np.unique(y_pred):
        cluster_mask = y_pred == cluster
        _, counts = np.unique(
            y_true[cluster_mask],
            return_counts=True,
        )
        total += int(counts.max())

    return float(total / len(y_true))


def normalized_entropy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute cluster-size-weighted entropy normalized by log2(C).

    Lower values indicate purer clusters.
    """
    number_of_classes = len(np.unique(y_true))

    if number_of_classes <= 1:
        return 0.0

    total_entropy = 0.0
    number_of_samples = len(y_true)

    for cluster in np.unique(y_pred):
        cluster_mask = y_pred == cluster
        cluster_labels = y_true[cluster_mask]

        _, counts = np.unique(
            cluster_labels,
            return_counts=True,
        )

        probabilities = counts / counts.sum()
        cluster_entropy = -np.sum(
            probabilities * np.log2(probabilities + 1e-12)
        )

        total_entropy += (
            len(cluster_labels) / number_of_samples
        ) * cluster_entropy

    maximum_entropy = np.log2(number_of_classes)

    return float(total_entropy / maximum_entropy)


def evaluate_clustering(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate all clustering-quality measures."""
    mapped_predictions = map_clusters_to_labels(
        y_true,
        y_pred,
    )

    return {
        "F1_macro": float(
            f1_score(
                y_true,
                mapped_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "F1_weighted": float(
            f1_score(
                y_true,
                mapped_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "Purity": purity_score(y_true, y_pred),
        "NMI": float(
            normalized_mutual_info_score(y_true, y_pred)
        ),
        "ARI": float(
            adjusted_rand_score(y_true, y_pred)
        ),
        "Entropy": normalized_entropy_score(
            y_true,
            y_pred,
        ),
        "Accuracy": clustering_accuracy(
            y_true,
            y_pred,
        ),
        "V_measure": float(
            v_measure_score(y_true, y_pred)
        ),
    }


def spherical_kmeans_fit_predict(
    matrix: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    """Run a reproducible Spherical K-Means implementation."""
    generator = np.random.default_rng(random_state)
    normalized_matrix = normalize(matrix, norm="l2")
    number_of_samples = normalized_matrix.shape[0]

    if n_clusters > number_of_samples:
        raise ValueError(
            "n_clusters cannot exceed the number of samples."
        )

    initial_indices = generator.choice(
        number_of_samples,
        size=n_clusters,
        replace=False,
    )

    centroids = normalized_matrix[initial_indices]
    labels = np.full(number_of_samples, -1, dtype=int)

    for _ in range(max_iter):
        previous_labels = labels.copy()

        similarities = normalized_matrix @ centroids.T
        labels = np.argmax(similarities, axis=1)

        updated_centroids = []

        for cluster_index in range(n_clusters):
            cluster_points = normalized_matrix[
                labels == cluster_index
            ]

            if len(cluster_points) == 0:
                centroid = normalized_matrix[
                    generator.integers(0, number_of_samples)
                ]
            else:
                centroid = cluster_points.mean(axis=0)
                centroid = centroid / (
                    np.linalg.norm(centroid) + 1e-12
                )

            updated_centroids.append(centroid)

        centroids = np.vstack(updated_centroids)

        if np.array_equal(labels, previous_labels):
            break

    return labels


def run_algorithm(
    algorithm: str,
    *,
    matrix: np.ndarray,
    spectral_matrix: np.ndarray,
    n_clusters: int,
    seed: int,
    settings: ClusteringSettings,
) -> np.ndarray:
    """Run one clustering algorithm."""
    if algorithm == "kmeansplusplus":
        return KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=seed,
            n_init=settings.n_init,
            max_iter=settings.max_iter,
        ).fit_predict(matrix)

    if algorithm == "spherical_kmeans":
        return spherical_kmeans_fit_predict(
            matrix,
            n_clusters=n_clusters,
            random_state=seed,
            max_iter=settings.max_iter,
        )

    if algorithm == "hac":
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=settings.hac_linkage,
        ).fit_predict(matrix)

    if algorithm == "spectral_coclustering":
        model = SpectralCoclustering(
            n_clusters=n_clusters,
            random_state=seed,
        )
        model.fit(spectral_matrix)
        return np.asarray(model.row_labels_)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_assignment_output(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    seed: int,
    algorithm: str,
) -> pd.DataFrame:
    """Create a compact cluster-assignment output dataframe."""
    text_column = find_first_existing_column(
        frame,
        ("text", "document"),
        required=False,
    )
    processed_column = find_first_existing_column(
        frame,
        ("ProcessedDocument", "processed_document"),
        required=False,
    )

    output = pd.DataFrame(
        {
            "id": frame["id"].values,
            "label": frame["label"].values,
            "cluster": predictions,
            "run": seed,
            "algorithm": algorithm,
        }
    )

    if text_column is not None:
        output.insert(
            0,
            "text",
            frame[text_column].values,
        )

    if processed_column is not None:
        insertion_position = (
            1 if "text" in output.columns else 0
        )
        output.insert(
            insertion_position,
            "ProcessedDocument",
            frame[processed_column].values,
        )

    return output


def build_result_record(
    *,
    embedding_file: Path,
    algorithm: str,
    seed: int,
    n_clusters: int,
    metrics: Dict[str, float],
) -> Dict[str, object]:
    """Create one long-format metric record."""
    record: Dict[str, object] = {
        "file_name": embedding_file.name,
        "dataset_embedding": embedding_file.stem,
        "algorithm": algorithm,
        "run": seed,
        "n_clusters": n_clusters,
    }

    record.update(metrics)
    return record


def seeds_for_algorithm(
    algorithm: str,
    seeds: Sequence[int],
) -> Tuple[int, ...]:
    """Use one deterministic run for HAC and all seeds otherwise."""
    if algorithm == "hac":
        return (int(seeds[0]),)

    return tuple(int(seed) for seed in seeds)


def process_embedding_file(
    embedding_file: Path,
    *,
    output_dir: Path,
    settings: ClusteringSettings,
) -> List[Dict[str, object]]:
    """Run all requested clustering algorithms for one embedding file."""
    LOGGER.info("Processing embedding file: %s", embedding_file)

    frame = pd.read_csv(embedding_file)
    embedding_columns = validate_embedding_file(frame)

    matrix = frame[embedding_columns].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Embedding matrix contains NaN or infinite values: "
            f"{embedding_file}"
        )

    # Spectral Co-Clustering requires a non-negative matrix.
    spectral_matrix = MinMaxScaler(
        feature_range=(0, 1)
    ).fit_transform(matrix)

    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(
        frame["label"].astype(str)
    )
    n_clusters = len(np.unique(y_true))

    LOGGER.info(
        "Samples=%d | Dimensions=%d | Clusters=%d",
        matrix.shape[0],
        matrix.shape[1],
        n_clusters,
    )

    result_records: List[Dict[str, object]] = []

    for algorithm in settings.algorithms:
        assignment_frames = []
        algorithm_seeds = seeds_for_algorithm(
            algorithm,
            settings.seeds,
        )

        for seed in algorithm_seeds:
            set_global_seed(seed)

            LOGGER.info(
                "Algorithm=%s | Seed=%d",
                algorithm,
                seed,
            )

            predictions = run_algorithm(
                algorithm,
                matrix=matrix,
                spectral_matrix=spectral_matrix,
                n_clusters=n_clusters,
                seed=seed,
                settings=settings,
            )

            metrics = evaluate_clustering(
                y_true,
                predictions,
            )

            result_records.append(
                build_result_record(
                    embedding_file=embedding_file,
                    algorithm=algorithm,
                    seed=seed,
                    n_clusters=n_clusters,
                    metrics=metrics,
                )
            )

            if settings.save_assignments:
                assignment_frames.append(
                    create_assignment_output(
                        frame,
                        predictions,
                        seed=seed,
                        algorithm=algorithm,
                    )
                )

        if settings.save_assignments and assignment_frames:
            assignment_path = (
                output_dir
                / "assignments"
                / (
                    f"{embedding_file.stem}_"
                    f"{algorithm}_all_runs.csv"
                )
            )

            if assignment_path.exists() and not settings.overwrite:
                LOGGER.info(
                    "Assignment output exists; keeping: %s",
                    assignment_path,
                )
            else:
                assignment_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                pd.concat(
                    assignment_frames,
                    ignore_index=True,
                ).to_csv(
                    assignment_path,
                    index=False,
                    encoding="utf-8-sig",
                )
                LOGGER.info(
                    "Saved assignments: %s",
                    assignment_path,
                )

    return result_records


def create_summary(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    """Create mean/std summaries by embedding file and algorithm."""
    summary = (
        all_results
        .groupby(
            ["dataset_embedding", "algorithm"],
            as_index=False,
        )[list(METRIC_COLUMNS)]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part)
        ).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    return summary


def create_ranking_summary(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create metric-specific and aggregate ranks."""
    ranking = summary.copy()

    higher_is_better = (
        "F1_macro_mean",
        "F1_weighted_mean",
        "Purity_mean",
        "NMI_mean",
        "ARI_mean",
        "Accuracy_mean",
        "V_measure_mean",
    )

    for metric in higher_is_better:
        ranking[f"{metric}_rank"] = ranking[metric].rank(
            method="min",
            ascending=False,
        )

    ranking["Entropy_mean_rank"] = ranking[
        "Entropy_mean"
    ].rank(
        method="min",
        ascending=True,
    )

    rank_columns = [
        column
        for column in ranking.columns
        if column.endswith("_rank")
    ]

    ranking["Total_Rank"] = ranking[
        rank_columns
    ].sum(axis=1)

    return ranking.sort_values(
        ["Total_Rank", "dataset_embedding", "algorithm"]
    ).reset_index(drop=True)


def discover_embedding_files(
    input_dir: Path,
    pattern: str,
    recursive: bool,
) -> List[Path]:
    """Find embedding CSV files."""
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    files = sorted(
        input_dir.rglob(pattern)
        if recursive
        else input_dir.glob(pattern)
    )

    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} were found in {input_dir}"
        )

    return files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run clustering algorithms and evaluate document embeddings."
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing embedding CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for assignments, metrics, and summaries.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Embedding-file glob pattern.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively below --input_dir.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=SUPPORTED_ALGORITHMS,
        default=list(SUPPORTED_ALGORITHMS),
        help="Clustering algorithms to execute.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help=(
            "Random seeds for stochastic algorithms. "
            "HAC uses only the first seed because it is deterministic."
        ),
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="Number of K-Means++ initializations.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum iterations for K-Means methods.",
    )
    parser.add_argument(
        "--hac_linkage",
        choices=("ward", "complete", "average", "single"),
        default="ward",
        help="HAC linkage criterion.",
    )
    parser.add_argument(
        "--no_assignments",
        action="store_true",
        help="Do not save document-level cluster assignments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing assignment outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete clustering and evaluation pipeline."""
    args = parse_args()
    configure_logging(args.verbose)

    settings = ClusteringSettings(
        algorithms=tuple(args.algorithms),
        seeds=tuple(args.seeds),
        n_init=args.n_init,
        max_iter=args.max_iter,
        hac_linkage=args.hac_linkage,
        save_assignments=not args.no_assignments,
        overwrite=args.overwrite,
    )
    settings.validate()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_files = discover_embedding_files(
        input_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    LOGGER.info(
        "Found %d embedding files.",
        len(embedding_files),
    )

    all_records: List[Dict[str, object]] = []

    for embedding_file in embedding_files:
        try:
            all_records.extend(
                process_embedding_file(
                    embedding_file,
                    output_dir=output_dir,
                    settings=settings,
                )
            )
        except Exception:
            LOGGER.exception(
                "Failed to process: %s",
                embedding_file,
            )
            raise

    if not all_records:
        raise RuntimeError(
            "No clustering results were generated."
        )

    all_results = pd.DataFrame(all_records)

    results_path = (
        output_dir
        / "all_clustering_run_results.csv"
    )
    all_results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = create_summary(all_results)
    summary_path = (
        output_dir
        / "summary_mean_std_results.csv"
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    ranking = create_ranking_summary(summary)
    ranking_path = (
        output_dir
        / "ranking_summary_results.csv"
    )
    ranking.to_csv(
        ranking_path,
        index=False,
        encoding="utf-8-sig",
    )

    statistical_path = (
        output_dir
        / "statistical_testing_ready_results.csv"
    )
    all_results.to_csv(
        statistical_path,
        index=False,
        encoding="utf-8-sig",
    )

    LOGGER.info("Saved all-run metrics: %s", results_path)
    LOGGER.info("Saved mean/std summary: %s", summary_path)
    LOGGER.info("Saved ranking summary: %s", ranking_path)
    LOGGER.info(
        "Saved statistical-analysis input: %s",
        statistical_path,
    )
    LOGGER.info(
        "Clustering evaluation completed successfully."
    )


if __name__ == "__main__":
    main()
