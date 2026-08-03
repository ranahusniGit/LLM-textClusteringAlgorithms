# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:00:15 2026

@author: User
"""

# -*- coding: utf-8 -*-
"""
03_generate_camelbert_embeddings.py

Generate document embeddings using CAMeLBERT.

This script is a lightweight wrapper around
common_embedding_utils.py.

Example
-------
Single file

python 03_generate_camelbert_embeddings.py ^
    --input data/processed/01_light_stem_full_document.csv ^
    --output outputs/01_light_stem_full_document_camelbert_embeddings.csv

Entire directory

python 03_generate_camelbert_embeddings.py ^
    --input data/processed ^
    --output outputs/camelbert ^
    --pattern "*.csv" ^
    --output_suffix "_camelbert_embeddings.csv"
"""

from common_embedding_utils import run_embedding_pipeline


CONFIG = {

    # ----------------------------------------------------------
    # Model Information
    # ----------------------------------------------------------

    "model_label": "CAMeLBERT",

    "model_checkpoint":
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",

    # ----------------------------------------------------------
    # Preprocessing
    # ----------------------------------------------------------

    # CAMeLBERT uses the common preprocessing pipeline.
    "preprocessing_strategy": "none",

    # ----------------------------------------------------------
    # Tokenization
    # ----------------------------------------------------------

    "tokenization_settings": {
        "padding": "max_length",
        "truncation": True,
    },

    # ----------------------------------------------------------
    # Embedding Extraction
    # ----------------------------------------------------------

    "max_sequence_length": 256,

    "hidden_layer": -1,

    "pooling_strategy": "mean",

    "trust_remote_code": True,
}


if __name__ == "__main__":
    run_embedding_pipeline(CONFIG)