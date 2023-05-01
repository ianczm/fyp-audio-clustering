# Helpers

This folder contains all the helper classes that abstract away
tedious low-level implementation details from the notebooks.


### Table of Contents

- [Constants](#constants)
- [Processors](#processors)
- [Query](#query)
- [Repositories](#repositories)
- [Visualisation](#visualisation)


---


## Constants

This module supplies mapping tables for chords and the default parameter
values for other custom helper classes.


## Processors

This module is split into two classes:


- `AudioDataProcessor` converts a raw `.mp3` file into an `AudioData` object which
  carries information about the waveform of that file. It allows the program to
  understand music as a 2D time-series array.


- `FeatureVectorProcessor` is responsible for converting `AudioData` objects to
  `FeatureVector` and `FeatureRepresentation` classes. This class performs feature
  extraction on the raw audio using libraries like `madmom` and `librosa`.


Please see [Models: Features](/src/models/README.md) at `/src/models` to understand more.


## Query

The `NearestNeighbourQuery` class is a simple abstraction of using `sklearn`'s
`NearestNeighbour` algorithm to obtain the nearest neighbours of a point in a
Pandas dataset. It takes in 3 initialisation parameters:

- Coordinates of all points in a dataset.
- The metadata that each point is associated to.
- The number of nearest neighbours to retrieve.

The user simply runs the `.search()` method after initialisation with a search
term to get back the nearest neighbours.


## Repositories

Repositories is concerned with data management and is split into two classes:

- `PandasAudioRepository` stores individual `FeatureVector`s and optionally
  `FeatureRepresentation`s into compressed pickles of Pandas Dataframes. It also 
  provides a method to load all playlists from the `/data/extracted` directory
  automatically for convenience.


- `AudioRepository` is a deprecated class that loads and stores pickles of individual
  `FeatureVector`s and `FeatureRepresentation`s which is highly inefficient and
  time-consuming.


## Visualisation

This module contains a `Visualiser` class along with several helper functions that
can be easily imported from notebooks. It provides a high-level API for generating
complex plots and subplots.

However, many methods and functions are still tightly coupled with its original
implementation in the notebooks from which they were extracted.


---

# Models and Features

Please see [Models: Features](/src/models/README.md) for more information.

