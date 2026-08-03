# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:03:11 2026

@author: User
"""

# -*- coding: utf-8 -*-
"""
05_generate_arabertv2_embeddings.py

Generate document embeddings using AraBERTv2.

Unlike the other embedding models, AraBERTv2 was pretrained using
official AraBERT segmented text. Therefore, this script automatically
applies the official AraBERT preprocessing and segmentation pipeline
implemented inside common_embedding_utils.py before tokenization.

This implementation follows the recommendation provided by the
official AraBERT documentation and improves the fairness and
reproducibility of the experimental comparison.

Example
-------
Single file

python 05_generate_arabertv2_embeddings.py ^
    --input data/processed/01_light_stem_full_document.csv ^
    --output outputs/01_light_stem_full_document_arabertv2_embeddings.csv

Entire directory

python 05_generate_arabertv2_embeddings.py ^
    --input data/processed ^
    --output outputs/arabertv2 ^
    --pattern "*.csv" ^
    --output_suffix "_arabertv2_embeddings.csv"
"""

from common_embedding_utils import run_embedding_pipeline


CONFIG = {

    # ==========================================================
    # Model Information
    # ==========================================================

    "model_label": "AraBERTv2",

    "model_checkpoint":
        "aubmindlab/bert-base-arabertv2",

    # ==========================================================
    # IMPORTANT
    # ==========================================================

    #
    # AraBERTv2 requires the official AraBERT preprocessing
    # (including segmentation) before tokenization.
    #
    # The shared utility automatically performs this step.
    #

    "preprocessing_strategy": "arabert",

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

    "max_sequence_length": 256,

    "hidden_layer": -1,

    "pooling_strategy": "mean",

    "trust_remote_code": True,
}


if __name__ == "__main__":

    run_embedding_pipeline(CONFIG)