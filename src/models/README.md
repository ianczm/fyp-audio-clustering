# Features

This module focuses on abstracting the many features and representations
extracted from raw audio into objects.

Features can be broken down into:

- [AudioData](#audiodata)
- [FeatureVector](#featurevector)
- [FeatureRepresentation](#featurerepresentation)

### Data Management Philosophy

Storing data as objects helps to:

- Leverage Python intellisense.
- Allow cleaner dot notation to access members.
- Easily convert objects into Dataframe rows via the `dataclass` `asdict()` method.
- Prevents numpy arrays from cluttering up the notebook namespace.

---

## AudioData

Contains song metadata and the waveform as a numpy array.

## FeatureVector

Stores a reference to the original `AudioData` and includes as members all the
columns or features that would be used in the actual Dataset. It represents one
row of the song dataset.

## FeatureRepresentation

During computation by `librosa` to extract features, intermediate representations
of the raw audio can be stored in this object for reuse and visualisation.
