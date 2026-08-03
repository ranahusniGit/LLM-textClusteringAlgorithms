# README

Comparative Evaluation of Arabic Large Language Models for Unsupervised Text Clustering

This repository contains the complete implementation required to reproduce the experiments reported in the manuscript.

Repository Structure
- preprocessing
- embedding_scripts
- run_all_embeddings.py
- clustering
- statistical_analysis
- data
- outputs

Workflow
1. Preprocess datasets.
2. Generate embeddings.
3. Run clustering.
4. Perform statistical analysis.
5. Reproduce tables and figures.

Embedding Models
- AraGPT
- AraBERT
- CAMeLBERT
- AraELECTRA
- AraBERTv2
- MultiMiniLM

Clustering Algorithms
- K-Means++
- Spherical K-Means
- HAC
- Spectral Co-Clustering

Evaluation Metrics
Accuracy, Macro F1, Weighted F1, Purity, NMI, ARI, V-measure, Entropy.

Datasets
Khaleej-2004, Morocco-2016, SANAD-AlKhaleej, Watan-2004, Aramed.
Please download the datasets from their original sources and place them under data/raw.

Reproducibility
- No hard-coded paths.
- Configurable command-line arguments.
- Fixed random seeds.
- HAC executed once.
- Other algorithms executed for 30 runs.
- AraBERTv2 uses the official preprocessing pipeline.
