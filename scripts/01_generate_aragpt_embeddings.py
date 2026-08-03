# -*- coding: utf-8 -*-
"""
01_generate_aragpt_embeddings.py

Generate document embeddings with the AraGPT2-base checkpoint.

This script is a lightweight model-specific wrapper around
``common_embedding_utils.py``. All input/output paths and runtime settings are
provided through command-line arguments; no local paths are hard-coded.

Model configuration
-------------------
- Manuscript label: AraGPT
- Checkpoint: aubmindlab/AraGPT2-base
- Architecture: decoder-only transformer
- Hidden representation: final hidden layer
- Pooling: last valid token
- Maximum sequence length: 256 tokens by default
- Padding: max_length
- Truncation: enabled
- Model-specific preprocessing: none

Example: one processed dataset
------------------------------
python 01_generate_aragpt_embeddings.py ^
    --input data/processed/khaleej/01_light_stem_full_document.csv ^
    --output outputs/embeddings/khaleej/01_light_stem_full_document_aragpt_embeddings.csv ^
    --text_column ProcessedDocument ^
    --batch_size 16 ^
    --device auto ^
    --seed 42

Example: all processed CSV files in one directory
-------------------------------------------------
python 01_generate_aragpt_embeddings.py ^
    --input data/processed/khaleej ^
    --output outputs/embeddings/khaleej/aragpt ^
    --pattern "*.csv" ^
    --output_suffix "_aragpt_embeddings.csv" ^
    --batch_size 16 ^
    --device auto ^
    --seed 42
"""

from __future__ import annotations

from common_embedding_utils import run_embedding_pipeline


CONFIG = {
    # Name used in logs and output documentation.
    "model_label": "AraGPT",

    # Hugging Face checkpoint used in the experiments.
    "model_checkpoint": "aubmindlab/AraGPT2-base",

    # AraGPT does not require the AraBERT segmentation pipeline.
    "preprocessing_strategy": "none",

    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True,
    },

    # Can be overridden from the command line using --max_length.
    "max_sequence_length": 256,

    # Final hidden layer.
    "hidden_layer": -1,

    # Decoder-only models do not provide a CLS representation.
    # The final non-padding token is used as the document vector.
    "pooling_strategy": "last_token",

    # Kept explicit for transparent model loading.
    "trust_remote_code": True,
}


if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)
