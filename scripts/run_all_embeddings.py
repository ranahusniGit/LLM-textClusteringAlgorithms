# -*- coding: utf-8 -*-
"""
run_all_embeddings.py

Run one or more embedding-generation scripts over all processed CSV files.

This script replaces the previous duplicated runners:
- runAllEmbeddingOnAllFilesFinal.py
- runAllEmbeddingOnAllFilesFinalUpdatedAfterR2.py

Features
--------
- No hard-coded local paths.
- Configurable input, output, and embedding-script folders.
- Select specific embeddings or run all available embeddings.
- Skips existing outputs unless --overwrite is supplied.
- Supports configurable batch size, device, seed, text column, and max length.
- Captures subprocess output and reports failures clearly.
- Uses consistent manuscript names.

Expected repository structure
-----------------------------
project_root/
├── run_all_embeddings.py
└── arabic_embedding_scripts/
    ├── common_embedding_utils.py
    ├── 01_generate_aragpt_embeddings.py
    ├── 02_generate_arabert_embeddings.py
    ├── 03_generate_camelbert_embeddings.py
    ├── 04_generate_araelectra_embeddings.py
    ├── 05_generate_arabertv2_embeddings.py
    └── 06_generate_multiminilm_embeddings.py

Examples
--------
Run all embeddings:

python run_all_embeddings.py ^
    --input_folder data/processed ^
    --output_folder outputs/embeddings ^
    --embedding_folder arabic_embedding_scripts ^
    --embeddings all ^
    --batch_size 16 ^
    --device auto ^
    --seed 42

Run selected embeddings only:

python run_all_embeddings.py ^
    --input_folder data/processed ^
    --output_folder outputs/embeddings ^
    --embedding_folder arabic_embedding_scripts ^
    --embeddings arabert arabertv2 multiminilm

Overwrite existing outputs:

python run_all_embeddings.py ^
    --input_folder data/processed ^
    --output_folder outputs/embeddings ^
    --embedding_folder arabic_embedding_scripts ^
    --embeddings all ^
    --overwrite
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


LOGGER = logging.getLogger("run_all_embeddings")


# ---------------------------------------------------------------------
# Available embedding wrappers
# ---------------------------------------------------------------------

EMBEDDING_SCRIPTS: Dict[str, Tuple[str, str]] = {
    "aragpt": (
        "01_generate_aragpt_embeddings.py",
        "aragpt",
    ),
    "arabert": (
        "02_generate_arabert_embeddings.py",
        "arabert",
    ),
    "camelbert": (
        "03_generate_camelbert_embeddings.py",
        "camelbert",
    ),
    "araelectra": (
        "04_generate_araelectra_embeddings.py",
        "araelectra",
    ),
    "arabertv2": (
        "05_generate_arabertv2_embeddings.py",
        "arabertv2",
    ),
    "multiminilm": (
        "06_generate_multiminilm_embeddings.py",
        "multiminilm",
    ),
}


@dataclass(frozen=True)
class RunnerSettings:
    """Execution settings for all embedding subprocesses."""

    batch_size: int
    device: str
    seed: int
    text_column: str
    max_length: int | None
    overwrite: bool
    continue_on_error: bool
    verbose_subprocess: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("max_length must be greater than zero.")


def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def discover_input_files(
    input_folder: Path,
    pattern: str,
    recursive: bool,
) -> List[Path]:
    """Discover processed CSV files."""
    input_folder = Path(input_folder)

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    files = sorted(
        input_folder.rglob(pattern)
        if recursive
        else input_folder.glob(pattern)
    )

    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern!r} found in {input_folder}"
        )

    return files


def resolve_embeddings(
    requested: Sequence[str],
) -> List[str]:
    """Resolve 'all' or validate selected embedding names."""
    normalized = [
        value.strip().lower()
        for value in requested
        if value.strip()
    ]

    if not normalized:
        raise ValueError("At least one embedding must be selected.")

    if "all" in normalized:
        return list(EMBEDDING_SCRIPTS.keys())

    invalid = [
        value
        for value in normalized
        if value not in EMBEDDING_SCRIPTS
    ]

    if invalid:
        raise ValueError(
            f"Unsupported embeddings: {invalid}. "
            f"Available values: {list(EMBEDDING_SCRIPTS)}"
        )

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(normalized))


def build_command(
    script_path: Path,
    input_file: Path,
    output_file: Path,
    settings: RunnerSettings,
) -> List[str]:
    """Build the embedding-wrapper command."""
    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--batch_size",
        str(settings.batch_size),
        "--device",
        settings.device,
        "--seed",
        str(settings.seed),
        "--text_column",
        settings.text_column,
    ]

    if settings.max_length is not None:
        command.extend(
            [
                "--max_length",
                str(settings.max_length),
            ]
        )

    if settings.overwrite:
        command.append("--overwrite")

    return command


def run_one_embedding(
    *,
    script_path: Path,
    input_file: Path,
    output_file: Path,
    settings: RunnerSettings,
) -> bool:
    """
    Run one embedding script.

    Returns
    -------
    bool
        True when successful, otherwise False.
    """
    command = build_command(
        script_path=script_path,
        input_file=input_file,
        output_file=output_file,
        settings=settings,
    )

    LOGGER.info("Running: %s", " ".join(command))

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if settings.verbose_subprocess and result.stdout:
        LOGGER.info(
            "Subprocess stdout for %s:\n%s",
            input_file.name,
            result.stdout,
        )

    if result.returncode == 0:
        LOGGER.info("Saved: %s", output_file)
        return True

    LOGGER.error(
        "Embedding generation failed.\n"
        "Input: %s\n"
        "Script: %s\n"
        "Command: %s\n"
        "Return code: %s\n"
        "STDOUT:\n%s\n"
        "STDERR:\n%s",
        input_file,
        script_path,
        " ".join(command),
        result.returncode,
        result.stdout,
        result.stderr,
    )

    return False


def run_all_embeddings(
    *,
    input_folder: Path,
    output_folder: Path,
    embedding_folder: Path,
    embeddings: Sequence[str],
    pattern: str,
    recursive: bool,
    settings: RunnerSettings,
) -> None:
    """Run selected embeddings over all discovered processed files."""
    settings.validate()

    input_files = discover_input_files(
        input_folder,
        pattern=pattern,
        recursive=recursive,
    )

    selected_embeddings = resolve_embeddings(embeddings)

    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_folder = Path(embedding_folder)

    LOGGER.info("Found %d input file(s).", len(input_files))
    LOGGER.info(
        "Selected embeddings: %s",
        ", ".join(selected_embeddings),
    )

    successes: List[str] = []
    skipped: List[str] = []
    failures: List[str] = []

    for input_file in input_files:
        LOGGER.info(
            "Processing input file: %s",
            input_file.name,
        )

        for embedding_key in selected_embeddings:
            script_name, output_label = EMBEDDING_SCRIPTS[
                embedding_key
            ]

            script_path = embedding_folder / script_name

            if not script_path.exists():
                message = (
                    f"Embedding script not found: {script_path}"
                )

                if settings.continue_on_error:
                    LOGGER.error(message)
                    failures.append(
                        f"{input_file.name} | {embedding_key}"
                    )
                    continue

                raise FileNotFoundError(message)

            output_file = (
                output_folder
                / (
                    f"{input_file.stem}_"
                    f"{output_label}_embeddings.csv"
                )
            )

            if output_file.exists() and not settings.overwrite:
                LOGGER.info(
                    "Skipping existing output: %s",
                    output_file.name,
                )
                skipped.append(
                    f"{input_file.name} | {embedding_key}"
                )
                continue

            successful = run_one_embedding(
                script_path=script_path,
                input_file=input_file,
                output_file=output_file,
                settings=settings,
            )

            result_name = (
                f"{input_file.name} | {embedding_key}"
            )

            if successful:
                successes.append(result_name)
            else:
                failures.append(result_name)

                if not settings.continue_on_error:
                    raise RuntimeError(
                        "Embedding execution stopped after failure: "
                        f"{result_name}"
                    )

    LOGGER.info("=" * 70)
    LOGGER.info("EMBEDDING EXECUTION SUMMARY")
    LOGGER.info("=" * 70)
    LOGGER.info("Successful: %d", len(successes))
    LOGGER.info("Skipped: %d", len(skipped))
    LOGGER.info("Failed: %d", len(failures))

    if failures:
        LOGGER.warning(
            "Failed jobs:\n%s",
            "\n".join(failures),
        )

    if failures and not settings.continue_on_error:
        raise RuntimeError(
            f"{len(failures)} embedding job(s) failed."
        )

    LOGGER.info(
        "All requested embedding jobs completed."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run selected embedding-generation scripts over all "
            "processed CSV files."
        )
    )

    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing processed CSV files.",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Folder where embedding CSV files will be saved.",
    )
    parser.add_argument(
        "--embedding_folder",
        default="arabic_embedding_scripts",
        help="Folder containing model-specific embedding scripts.",
    )
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=["all"],
        help=(
            "Embeddings to execute. Available: "
            "aragpt arabert camelbert araelectra arabertv2 "
            "multiminilm all"
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Processed-file glob pattern.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively below --input_folder.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Embedding extraction batch size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to each embedding script.",
    )
    parser.add_argument(
        "--text_column",
        default="ProcessedDocument",
        help="Processed-text column passed to embedding scripts.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help=(
            "Optional maximum sequence-length override. "
            "Model default is used when omitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embedding files.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing after a failed embedding job.",
    )
    parser.add_argument(
        "--verbose_subprocess",
        action="store_true",
        help="Display successful embedding-script stdout.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose runner logging.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    configure_logging(args.verbose)

    settings = RunnerSettings(
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        text_column=args.text_column,
        max_length=args.max_length,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        verbose_subprocess=args.verbose_subprocess,
    )

    run_all_embeddings(
        input_folder=Path(args.input_folder),
        output_folder=Path(args.output_folder),
        embedding_folder=Path(args.embedding_folder),
        embeddings=args.embeddings,
        pattern=args.pattern,
        recursive=args.recursive,
        settings=settings,
    )


if __name__ == "__main__":
    main()
