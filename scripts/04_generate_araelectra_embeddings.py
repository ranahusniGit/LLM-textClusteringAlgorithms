# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:01:43 2026

@author: User
"""

# -*- coding: utf-8 -*-
"""
04_generate_araelectra_embeddings.py

Generate document embeddings using AraELECTRA.

This script is a lightweight wrapper around
common_embedding_utils.py and contains only the
model-specific configuration.

Example
-------
Single file

python 04_generate_araelectra_embeddings.py ^
    --input data/processed/01_light_stem_full_document.csv ^
    --output outputs/01_light_stem_full_document_araelectra_embeddings.csv

Entire directory

python 04_generate_araelectra_embeddings.py ^
    --input data/processed ^
    --output outputs/araelectra ^
    --pattern "*.csv" ^
    --output_suffix "_araelectra_embeddings.csv"
"""

from common_embedding_utils import run_embedding_pipeline


CONFIG = {

    # ==========================================================
    # Model Information
    # ==========================================================

    "model_label": "AraELECTRA",

    "model_checkpoint":
        "aubmindlab/AraELECTRA-base-discriminator",

    # ==========================================================
    # Preprocessing
    # ==========================================================

    # AraELECTRA uses the common preprocessing pipeline.
    "preprocessing_strategy": "none",

    # ==========================================================
    # Tokenization
    # ==========================================================

    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True,
    },

    # ==========================================================
    # Embedding Extraction
    # ==========================================================

    # Default maximum sequence length.
    # Can be overridden from the command line using:
    #
    # --max_length
    #
    "max_sequence_length": 256,

    # Last hidden layer.
    "hidden_layer": -1,

    # Mean pooling over all non-padding tokens.
    "pooling_strategy": "mean",

    # Explicitly retained for transparency.
    "trust_remote_code": True,
}


if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)