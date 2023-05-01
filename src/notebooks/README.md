# Notebooks

This is the heart of the program. All code related to the Methodology section
of the Final Year Project Report can be found in these notebooks.

Jupyter Notebooks were used to explore and visualise the song dataset. Most figures
generated will also be saved to the [`/data/temp/images` folder](/data/temp/README.md).
(Not to be confused with the [`/figures` folder](/figures/README.md))


### Table of Contents

Notebooks must be run in this order if they are being run for the first time. This
is to generate all the needed `temp` folders.

1. Feature Extraction
2. Data Exploration
3. MDS Embedding
4. PCA Dimensionality Reduction
5. T-SNE Hyper-Parameters
6. T-SNE Visualisation and User Input


---


## Description

The Methodology and results obtained from these notebooks are described extensively
in the Final Year Project Report.

Most operations in the notebooks rely on:

- `/src/helpers` module for a high-level API for operations and graphing.

- `config.py`, a configuration file specifically for use among notebooks to
indicate destinations for saving temporary data and figures.

- `/data/temp` for storing temporary data that allows multiple notebooks to be used
in a pipeline.

- `/data/extracted` for reading in songs to create the dataset.

