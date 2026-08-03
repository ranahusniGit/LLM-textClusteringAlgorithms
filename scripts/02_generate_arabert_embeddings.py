# -*- coding: utf-8 -*-
"""
02_generate_arabert_embeddings.py

Generate document embeddings with the AraBERT v0.2 checkpoint.

This script is a lightweight model-specific wrapper around
``common_embedding_utils.py``. All paths and runtime settings are supplied
through command-line arguments; no machine-specific paths are hard-coded.

Model configuration
-------------------
- Manuscript label: AraBERT
- Checkpoint: aubmindlab/bert-base-arabertv02
- Architecture: encoder transformer
- Hidden representation: final hidden layer
- Pooling: mean over non-padding tokens
- Maximum sequence length: 256 tokens by default
- Padding: max_length
- Truncation: enabled
- Model-specific preprocessing: none

Example: one processed dataset
------------------------------
python 02_generate_arabert_embeddings.py ^
    --input data/processed/khaleej/01_light_stem_full_document.csv ^
    --output outputs/embeddings/khaleej/01_light_stem_full_document_arabert_embeddings.csv ^
    --text_column ProcessedDocument ^
    --batch_size 16 ^
    --device auto ^
    --seed 42

Example: all processed CSV files in one directory
-------------------------------------------------
python 02_generate_arabert_embeddings.py ^
    --input data/processed/khaleej ^
    --output outputs/embeddings/khaleej/arabert ^
    --pattern "*.csv" ^
    --output_suffix "_arabert_embeddings.csv" ^
    --batch_size 16 ^
    --device auto ^
    --seed 42
"""

from __future__ import annotations

from common_embedding_utils import run_embedding_pipeline


CONFIG = {
    # Name used in logs and output documentation.
    "model_label": "AraBERT",

    # Hugging Face checkpoint used in the experiments.
    "model_checkpoint": "aubmindlab/bert-base-arabertv02",

    # The common LS/WS preprocessing output is passed directly to this model.
    "preprocessing_strategy": "none",

    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True,
    },

    # Can be overridden from the command line using --max_length.
    "max_sequence_length": 256,

    # Final hidden layer.
    "hidden_layer": -1,

    # Mean pooling over non-padding token representations.
    "pooling_strategy": "mean",

    # Kept explicit for transparent model loading.
    "trust_remote_code": True,
}


if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)
