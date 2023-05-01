# Clustering Music with Spectral and Harmonic Features

Submitted in partial fulfilment of the requirements for the award of: 
Bachelor of Science in Computer Science with Artificial Intelligence (Hons),
The University of Nottingham


### Student Details

| Field        | Details             |
|--------------|---------------------|
| Student Name | Ian Chong Zhen Ming |
| Student ID   | 20313229            |
| Supervisor   | Dr Radu Muschevici  |
| Date         | 1 May, 2023         |


### Table of Contents

- [Dissertation Abstract](#dissertation-abstract)
- [Installation](#installation)
- [Usage](#usage)

---


## Dissertation Abstract

_Taken from the Final Year Project Report._

This project introduces music clustering as an unsupervised method of generating music recommendations,
where songs are represented by spectral and harmonic features extracted directly from its audio,
aiming to produce highly similar recommendations as the reference song.

Data flows through a pipeline comprised of Multidimensional Scaling and Principal Component Analysis
for dimensionality reduction, the Gaussian Mixture Model for clustering as well as the t-Distributed
Stochastic Neighbour Embedding and k Nearest Neighbours for visualisation and recommendation generation.

This project demonstrates that clustering is indeed a viable option for generating recommendations,
grouping a dataset of 425 songs into 8 clusters that closely represent the 8 playlists that the songs were
originally selected from. This gives hope that with a better feature set, it would be possible to generate
consistent and highly similar recommendations, allowing users to discover new music without having to
condense their search result into human-readable words that might dilute the musical meaning behind
the search.


---


## Installation

This project depends on Python 3.10.9 and other packages listed
in the [`requirements.txt` file](requirements.txt).


### Install Python

If you do not have Python installed on your machine, you can download and install it from the official Python website:

1. Go to https://www.python.org/downloads/
2. Download the appropriate installer for your operating system (Windows, macOS, or Linux)
3. Run the installer and follow the prompts to complete the installation process.


### Install Anaconda

If you prefer to use Anaconda, a popular data science platform that includes Python and many pre-installed packages, you can download and install it from the official Anaconda website:

1. Go to https://www.anaconda.com/download/
2. Download the appropriate installer for your operating system (Windows, macOS, or Linux)
3. Run the installer and follow the prompts to complete the installation process.


### Clone the Repository

Clone this repository onto your local filesystem and then set it as the active directory.

```sh
git clone --recurse-submodules https://github.com/ianczm/fyp-audio-clustering.git
cd fyp-audio-clustering
```


### Set Up the Virtual Environment

Python comes with a built-in module called `venv` that allows
you to create and manage virtual environments.

```sh
python3 -m venv <env-name>
source <env-name>/bin/activate
```

If you prefer to use the Anaconda distribution, you can create
a new virtual environment using `conda`.

```sh
conda create -n <env-name> python=3.10.9
conda activate <env-name>
```


### Install Dependencies

Install all required dependencies.

```sh
pip install -r requirements.txt
```


---


## Usage

Please ensure sure your virtual environment is active before running this project.
It is recommended to use a Jupyter Notebook-supported IDE.


### Generate Your Dataset

This repository comes with pre-computed feature vectors stored in `/data/extracted`,
so this step is not necessary unless you want to extend the existing dataset with
your own Spotify playlist.

With your virtual environment still active, set `/scripts/spotify-downloader` as
your active directory and run `script.py` with Python.

```sh
cd /scripts/spotify-downloader
python script.py
```

Follow the prompts and set the output directory to `/data/raw/<playlist-name>`.

Once the downloads are completed, go to the `/scripts` directory and run the
`processAllPlaylists.py` script.

```sh
cd ../
python processAllPlaylists.py
```

Follow the prompts. Your playlist of `.mp3` will be processed and the new dataset
will be generated at `/data/extracted`.


### Run the Notebooks

If this is the first time running this program, navigate to the `/src/notebooks` folder
and run all notebooks in order. [See more details here](/src/notebooks/README.md).
Please ensure that each notebook completely finishes running before moving on to the next.


### Query Songs

Once all notebooks have completed execution, run the [`5-tsne-visualization-and-user-input`
notebook](/src/notebooks/5-tsne-visualization-and-user-input.ipynb).

On Input Line 11, there is a variable named `search_term`. Change the value of this
variable according to the name of the song you would like to search for, then run the
3 succeeding lines again to retrieve the k nearest neighbours as recommendations.

You can change the value of k on Input Line 10, which is set to a default of 6, where
the 0th nearest neighbour is the search term itself.


### View Figures

As each notebook is run, figures are generated and saved to `/temp/images` for further
analysis if needed. The `/figures` folder gives examples of figures that have been
generated before. This folder should not be modified and only serves as demonstration.
