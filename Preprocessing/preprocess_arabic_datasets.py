# -*- coding: utf-8 -*-
"""
preprocess_arabic_datasets.py

Reproducible Arabic preprocessing pipeline used to generate six document
representations for clustering experiments:

1. Light stemming + full document
2. Light stemming + first half
3. Light stemming + first 80 words
4. Without stemming + full document
5. Without stemming + first half
6. Without stemming + first 80 words

The script supports CSV and Excel inputs, configurable column names, external
stopword files, relative paths, structured logging, and command-line execution.

Example
-------
python preprocess_arabic_datasets.py ^
    --input data/raw/KhaleejDataSet.xls ^
    --stopwords data/resources/RemovedKeywords.xls ^
    --output_dir data/processed/khaleej ^
    --text_column text ^
    --output_text_column ProcessedDocument ^
    --output_format csv
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

import emoji
import pandas as pd
from tashaphyne.stemming import ArabicLightStemmer


LOGGER = logging.getLogger("arabic_preprocessing")


# ---------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------

_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
_TATWEEL_RE = re.compile(r"\u0640+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
_DIGITS_RE = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]+")
_PUNCT_RE = re.compile(
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~…–—•·
        \u0609\u060A\u060C\u061B\u061F\u066A\u066B\u066C\u06D4
        \uFE50-\uFE52\uFE54-\uFE57\uFE5F-\uFE63\uFE68\u066D]""",
    re.VERBOSE,
)
_ARABIC_CHAR_RE = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    r"\uFB50-\uFDFF\uFE70-\uFEFF\s]"
)
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{1,}")

_STEMMER = ArabicLightStemmer()


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration controlling Arabic text normalization."""

    keep_hashtags: bool = False
    keep_mentions: bool = False
    replace_digits: bool = True
    digit_token: str = "NUM"
    remove_stopwords: bool = True
    map_ta_marbuta_to_ha: bool = False
    map_alef_maqsurah_to_ya: bool = True
    map_hamza_variants: bool = True
    keep_only_arabic_chars: bool = True
    min_token_length: int = 2

    def validate(self) -> None:
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be at least 1.")


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------

def read_tabular_file(
    path: Path,
    *,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Read a CSV or Excel file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)

    raise ValueError(
        f"Unsupported file type {suffix!r}. Use CSV, XLSX, or XLS."
    )


def read_stopwords(
    stopwords_path: Path,
    *,
    stopword_column: Optional[str] = None,
    sheet_name: str | int = 0,
) -> Set[str]:
    """
    Read stopwords or removed keywords from TXT, CSV, XLSX, or XLS.
    """
    stopwords_path = Path(stopwords_path)

    if not stopwords_path.exists():
        raise FileNotFoundError(
            f"Stopword file not found: {stopwords_path}"
        )

    suffix = stopwords_path.suffix.lower()

    if suffix == ".txt":
        with stopwords_path.open("r", encoding="utf-8") as handle:
            return {
                line.strip()
                for line in handle
                if line.strip()
            }

    frame = read_tabular_file(
        stopwords_path,
        sheet_name=sheet_name,
    )

    if frame.empty:
        return set()

    selected_column = (
        stopword_column
        if stopword_column is not None
        else str(frame.columns[0])
    )

    if selected_column not in frame.columns:
        raise ValueError(
            f"Stopword column {selected_column!r} was not found. "
            f"Available columns: {list(frame.columns)}"
        )

    values = (
        frame[selected_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return {
        value
        for value in values
        if value
    }


# ---------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------

def light_stem_text(text: str) -> str:
    """Apply Tashaphyne light stemming token by token."""
    if not isinstance(text, str):
        return ""

    stemmed_tokens = []

    for token in text.split():
        _STEMMER.light_stem(token)
        stemmed_tokens.append(_STEMMER.get_stem())

    return " ".join(stemmed_tokens)


def normalize_arabic_text(
    text: str,
    *,
    config: PreprocessingConfig,
) -> str:
    """Clean and normalize Arabic text before lexical filtering."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)

    if not config.keep_mentions:
        text = _MENTION_RE.sub(" ", text)

    if not config.keep_hashtags:
        text = _HASHTAG_RE.sub(" ", text)

    text = emoji.replace_emoji(text, replace=" ")
    text = _DIACRITICS_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)

    # Collapse elongated forms, e.g. جمييييل -> جميل.
    text = _REPEATED_CHAR_RE.sub(r"\1", text)

    # Normalize Arabic letter variants.
    text = re.sub(r"[إأآا]", "ا", text)

    if config.map_alef_maqsurah_to_ya:
        text = text.replace("ى", "ي")

    if config.map_ta_marbuta_to_ha:
        text = text.replace("ة", "ه")

    if config.map_hamza_variants:
        text = (
            text.replace("ؤ", "و")
            .replace("ئ", "ي")
            .replace("ء", "")
        )

    if config.replace_digits:
        text = _DIGITS_RE.sub(
            f" {config.digit_token} ",
            text,
        )

    text = _PUNCT_RE.sub(" ", text)

    if config.keep_only_arabic_chars:
        text = _ARABIC_CHAR_RE.sub(" ", text)

    return _WHITESPACE_RE.sub(" ", text).strip()


def filter_tokens(
    text: str,
    *,
    stopwords: Set[str],
    config: PreprocessingConfig,
) -> str:
    """Remove stopwords and short tokens."""
    tokens = text.split()

    if config.remove_stopwords:
        tokens = [
            token
            for token in tokens
            if token not in stopwords
        ]

    tokens = [
        token
        for token in tokens
        if len(token) >= config.min_token_length
    ]

    return " ".join(tokens)


def clean_arabic_text(
    text: str,
    *,
    stopwords: Set[str],
    config: PreprocessingConfig,
    apply_light_stemming: bool,
) -> str:
    """Run the complete Arabic cleaning and morphology pipeline."""
    normalized = normalize_arabic_text(
        text,
        config=config,
    )

    filtered = filter_tokens(
        normalized,
        stopwords=stopwords,
        config=config,
    )

    if apply_light_stemming:
        return light_stem_text(filtered)

    return filtered


# ---------------------------------------------------------------------
# Document-length variants
# ---------------------------------------------------------------------

def first_half_text(text: str) -> str:
    """Return the first half of a document based on word count."""
    if not isinstance(text, str):
        return ""

    words = text.split()

    if not words:
        return ""

    half_size = max(1, len(words) // 2)
    return " ".join(words[:half_size])


def first_n_words(text: str, n_words: int) -> str:
    """Return the first ``n_words`` tokens from a document."""
    if not isinstance(text, str):
        return ""

    if n_words <= 0:
        raise ValueError("n_words must be greater than zero.")

    return " ".join(text.split()[:n_words])


def select_document_part(
    text: str,
    *,
    document_part: str,
    first_n: int,
) -> str:
    """Select the requested document-length representation."""
    if document_part == "full":
        return text

    if document_part == "first_half":
        return first_half_text(text)

    if document_part == "first_n_words":
        return first_n_words(text, first_n)

    raise ValueError(
        "document_part must be one of: "
        "'full', 'first_half', or 'first_n_words'."
    )


# ---------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------

def generate_one_version(
    frame: pd.DataFrame,
    *,
    text_column: str,
    output_text_column: str,
    stopwords: Set[str],
    config: PreprocessingConfig,
    apply_light_stemming: bool,
    document_part: str,
    first_n: int,
) -> pd.DataFrame:
    """Generate one preprocessing/document-length configuration."""
    if text_column not in frame.columns:
        raise ValueError(
            f"Text column {text_column!r} was not found. "
            f"Available columns: {list(frame.columns)}"
        )

    output = frame.copy()

    def process_document(value: object) -> str:
        text = "" if pd.isna(value) else str(value)

        # This order preserves the original experiment implementation:
        # select the document portion first, then clean/filter/stem it.
        selected = select_document_part(
            text,
            document_part=document_part,
            first_n=first_n,
        )

        return clean_arabic_text(
            selected,
            stopwords=stopwords,
            config=config,
            apply_light_stemming=apply_light_stemming,
        )

    output[output_text_column] = (
        output[text_column]
        .apply(process_document)
    )

    return output


def save_tabular_file(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a dataframe as CSV or Excel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix == ".csv":
        frame.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        return

    if suffix in {".xlsx", ".xls"}:
        frame.to_excel(
            output_path,
            index=False,
        )
        return

    raise ValueError(
        f"Unsupported output extension {suffix!r}. "
        "Use .csv, .xlsx, or .xls."
    )


def generate_six_files(
    *,
    input_path: Path,
    output_dir: Path,
    stopwords_path: Path,
    text_column: str,
    output_text_column: str,
    output_format: str,
    stopword_column: Optional[str] = None,
    input_sheet_name: str | int = 0,
    stopword_sheet_name: str | int = 0,
    first_n: int = 80,
    config: Optional[PreprocessingConfig] = None,
    overwrite: bool = False,
) -> Sequence[Path]:
    """Generate and save the six preprocessing configurations."""
    config = config or PreprocessingConfig()
    config.validate()

    frame = read_tabular_file(
        Path(input_path),
        sheet_name=input_sheet_name,
    )

    stopwords = read_stopwords(
        Path(stopwords_path),
        stopword_column=stopword_column,
        sheet_name=stopword_sheet_name,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extension = (
        ".xlsx"
        if output_format.lower() == "xlsx"
        else ".csv"
    )

    versions: Sequence[Tuple[str, bool, str]] = (
        ("01_light_stem_full_document", True, "full"),
        ("02_light_stem_first_half_document", True, "first_half"),
        (f"03_light_stem_first_{first_n}_words", True, "first_n_words"),
        ("04_without_stem_full_document", False, "full"),
        ("05_without_stem_first_half_document", False, "first_half"),
        (f"06_without_stem_first_{first_n}_words", False, "first_n_words"),
    )

    generated_paths = []

    for filename, use_stemming, document_part in versions:
        output_path = output_dir / f"{filename}{extension}"

        if output_path.exists() and not overwrite:
            LOGGER.info("Skipping existing output: %s", output_path)
            generated_paths.append(output_path)
            continue

        LOGGER.info("Generating: %s", filename)

        processed = generate_one_version(
            frame,
            text_column=text_column,
            output_text_column=output_text_column,
            stopwords=stopwords,
            config=config,
            apply_light_stemming=use_stemming,
            document_part=document_part,
            first_n=first_n,
        )

        save_tabular_file(
            processed,
            output_path,
        )

        LOGGER.info("Saved: %s", output_path)
        generated_paths.append(output_path)

    LOGGER.info(
        "Generated %d preprocessing files.",
        len(generated_paths),
    )

    return generated_paths


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_sheet_name(value: str) -> str | int:
    """Interpret numeric sheet values as integers."""
    try:
        return int(value)
    except ValueError:
        return value


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate six Arabic preprocessing configurations "
            "with/without light stemming and three document lengths."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV, XLSX, or XLS dataset.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where processed files will be saved.",
    )
    parser.add_argument(
        "--stopwords",
        required=True,
        help="TXT, CSV, XLSX, or XLS stopword/keyword file.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        help="Column containing the original document text.",
    )
    parser.add_argument(
        "--output_text_column",
        default="ProcessedDocument",
        help="Name of the generated processed-text column.",
    )
    parser.add_argument(
        "--stopword_column",
        default=None,
        help=(
            "Stopword column for CSV/Excel resources. "
            "The first column is used when omitted."
        ),
    )
    parser.add_argument(
        "--output_format",
        choices=("csv", "xlsx"),
        default="csv",
        help="Output file format.",
    )
    parser.add_argument(
        "--input_sheet",
        type=parse_sheet_name,
        default=0,
        help="Input Excel sheet name or zero-based index.",
    )
    parser.add_argument(
        "--stopword_sheet",
        type=parse_sheet_name,
        default=0,
        help="Stopword Excel sheet name or zero-based index.",
    )
    parser.add_argument(
        "--first_n_words",
        type=int,
        default=80,
        help="Number of words used in the shortened representation.",
    )
    parser.add_argument(
        "--min_token_length",
        type=int,
        default=2,
        help="Minimum retained token length.",
    )
    parser.add_argument(
        "--keep_hashtags",
        action="store_true",
        help="Preserve hashtags.",
    )
    parser.add_argument(
        "--keep_mentions",
        action="store_true",
        help="Preserve user mentions.",
    )
    parser.add_argument(
        "--keep_digits",
        action="store_true",
        help="Preserve digits instead of replacing them.",
    )
    parser.add_argument(
        "--digit_token",
        default="NUM",
        help="Replacement token used for digits.",
    )
    parser.add_argument(
        "--keep_stopwords",
        action="store_true",
        help="Do not remove stopwords.",
    )
    parser.add_argument(
        "--map_ta_marbuta_to_ha",
        action="store_true",
        help="Normalize ta marbuta to ha.",
    )
    parser.add_argument(
        "--keep_alef_maqsurah",
        action="store_true",
        help="Do not normalize alef maqsura to ya.",
    )
    parser.add_argument(
        "--keep_hamza_variants",
        action="store_true",
        help="Do not normalize hamza variants.",
    )
    parser.add_argument(
        "--allow_non_arabic",
        action="store_true",
        help="Preserve non-Arabic characters after cleaning.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)

    config = PreprocessingConfig(
        keep_hashtags=args.keep_hashtags,
        keep_mentions=args.keep_mentions,
        replace_digits=not args.keep_digits,
        digit_token=args.digit_token,
        remove_stopwords=not args.keep_stopwords,
        map_ta_marbuta_to_ha=args.map_ta_marbuta_to_ha,
        map_alef_maqsurah_to_ya=not args.keep_alef_maqsurah,
        map_hamza_variants=not args.keep_hamza_variants,
        keep_only_arabic_chars=not args.allow_non_arabic,
        min_token_length=args.min_token_length,
    )

    generate_six_files(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        stopwords_path=Path(args.stopwords),
        text_column=args.text_column,
        output_text_column=args.output_text_column,
        output_format=args.output_format,
        stopword_column=args.stopword_column,
        input_sheet_name=args.input_sheet,
        stopword_sheet_name=args.stopword_sheet,
        first_n=args.first_n_words,
        config=config,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
