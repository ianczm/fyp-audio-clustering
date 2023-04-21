import os
import pickle

import bz2file as bz2
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from glob import glob
from pathlib import Path

import pandas as pd

from src.models import *


class AbstractIO:
    @staticmethod
    def save(path: str, data) -> None:
        pass

    @staticmethod
    def load(path: str) -> any:
        pass

    @staticmethod
    def file_ext() -> str:
        pass


class CompressIO(AbstractIO):
    @staticmethod
    def save(path: str, data):
        with bz2.BZ2File(path + CompressIO.file_ext(), 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str):
        with bz2.BZ2File(path, 'rb') as data:
            return pickle.load(data)
        
    @staticmethod
    def file_ext() -> str:
        return '.pkl.pbz2'


class PickleIO(AbstractIO):
    @staticmethod
    def save(path: str, data):
        with open(path + PickleIO.file_ext(), 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str):
        with open(path, 'rb') as data:
            return pickle.load(data)
        
    @staticmethod
    def file_ext() -> str:
        return '.pkl'


class PandasAudioRepository:

    @staticmethod
    def load_all_feature_datasets(
            extracted_directory: str
    ):
        def get_feature_paths(extracted_directory: str):
            folders = glob(extracted_directory + '/*')
            results = []
            for folder in folders:
                for item in glob(folder + '/*'):
                    if '.features' in item:
                        results.append(item)
            return results

        def load_features(extracted_directory: str):
            feature_paths = get_feature_paths(extracted_directory)
            dataframes = []
            for path in feature_paths:
                dataframes.append(pd.read_pickle(path, compression='bz2'))
            return pd.concat(dataframes, axis=0)

        return load_features(extracted_directory)

    @staticmethod
    def store_datasets(
            directory: str,
            processed_audio: list[tuple[FeatureVector, FeatureRepresentation]]
    ):
        name = Path(directory).name
        vectors = [v for v, _ in processed_audio]
        representations = [r for _, r in processed_audio]

        dataset = PandasAudioRepository.store_feature_dataset(
            directory,
            name,
            vectors
        )

        PandasAudioRepository.store_repr_dataset(
            directory,
            name,
            dataset[['song_name', 'artist', 'playlist']],
            representations
        )

    @staticmethod
    def store_feature_dataset(
            directory: str,
            name: str,
            vectors: list[FeatureVector]
    ):
        dataset = pd.DataFrame([v.as_dict() | {'playlist': name} for v in vectors])
        dataset.to_pickle(str(Path(directory, name + '.features.pkl.pbz2')), compression='bz2')
        return dataset

    @staticmethod
    def store_repr_dataset(
            directory: str,
            name: str,
            metadata: pd.DataFrame,
            representations: list[FeatureRepresentation]
    ):
        representations_dataset = pd.DataFrame([r.as_dict() for r in representations])
        dataset = pd.concat([metadata, representations_dataset], axis=1)
        dataset.to_pickle(str(Path(directory, name + '.representations.pkl.pbz2')), compression='bz2')
        return dataset


class AudioRepository:

    io: AbstractIO = CompressIO()

    # Todo: Store vectors and representations separately for better memory management
    @staticmethod
    def store_one_processed_audio(directory: str, processed_audio: tuple[FeatureVector, FeatureRepresentation]):
        vector, _ = processed_audio
        directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        AudioRepository.io.save(f'{directory}/{vector.audio.name}', processed_audio)

    @staticmethod
    def load_one_processed_audio(file: str):
        loaded_object: tuple[FeatureVector, FeatureRepresentation] = AudioRepository.io.load(file)
        name = Path(file).parent.name
        loaded_object[0].audio.playlist = name
        return loaded_object

    # dir_start: which index to start from in each directory
    # dir_limit: how many songs to extract per directory
    @staticmethod
    def load_processed_audio(
            directories: list[str],
            dir_start: int = 0,
            dir_limit: int = 0
    ):
        processed_audio_futures: list[Future[tuple[FeatureVector, FeatureRepresentation]]] = []

        with ProcessPoolExecutor() as ec:
            for directory in directories:
                if dir_limit > 0:
                    files = glob(directory + '/*' + AudioRepository.io.file_ext())[dir_start:dir_start + dir_limit]
                else:
                    files = glob(directory + '/*' + AudioRepository.io.file_ext())
                for file in files:
                    if '.features' not in file and '.representations' not in file:
                        future = ec.submit(AudioRepository.load_one_processed_audio, file)
                        processed_audio_futures.append(future)
            results = [future.result() for future in as_completed(processed_audio_futures)]

        return results
