# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:05:37 2026

@author: User
"""

# -*- coding: utf-8 -*-
"""
06_generate_multiminilm_embeddings.py

Generate document embeddings using Multilingual MiniLM.

This script is a lightweight wrapper around
common_embedding_utils.py.

The model corresponds to the sentence-transformer
"paraphrase-multilingual-MiniLM-L12-v2", which is referred to
throughout the manuscript as MultiMiniLM.

Example
-------
Single file

python 06_generate_multiminilm_embeddings.py ^
    --input data/processed/01_light_stem_full_document.csv ^
    --output outputs/01_light_stem_full_document_multiminilm_embeddings.csv

Entire directory

python 06_generate_multiminilm_embeddings.py ^
    --input data/processed ^
    --output outputs/multiminilm ^
    --pattern "*.csv" ^
    --output_suffix "_multiminilm_embeddings.csv"
"""

from common_embedding_utils import run_embedding_pipeline


CONFIG = {

    # ==========================================================
    # Model Information
    # ==========================================================

    # Name used throughout the manuscript.
    "model_label": "MultiMiniLM",

    # Official Hugging Face checkpoint.
    "model_checkpoint":
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

    # ==========================================================
    # Preprocessing
    # ==========================================================

    # MultiMiniLM uses the common preprocessing pipeline.
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
    # Can be overridden using --max_length.
    "max_sequence_length": 256,

    # Last hidden layer.
    "hidden_layer": -1,

    # Mean pooling over non-padding tokens.
    "pooling_strategy": "mean",

    # Explicit for transparency.
    "trust_remote_code": True,
}


if __name__ == "__main__":

    run_embedding_pipeline(CONFIG)