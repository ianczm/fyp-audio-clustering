import os
import pickle
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from glob import glob
from pathlib import Path

from src.models import *


class AudioRepository:
    @staticmethod
    def store_processed_audio(directory: str, processed_audio: tuple[FeatureVector, FeatureRepresentation]):
        vector, _ = processed_audio
        directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(f'{directory}/{vector.audio.name}.data', 'wb') as f:
            pickle.dump(processed_audio, f)

    # dir_start: which index to start from in each directory
    # dir_limit: how many songs to extract per directory
    # Todo: store and load single files only, let script handle multiprocessing
    @staticmethod
    def load_processed_audio(directories: list[str], dir_start: int = 0, dir_limit: int = 0):
        processed_audio_futures: list[Future[tuple[FeatureVector, FeatureRepresentation]]] = []

        with ProcessPoolExecutor() as ec:
            for directory in directories:
                if dir_limit > 0:
                    files = glob(directory + '/*.data')[dir_start:dir_start + dir_limit]
                else:
                    files = glob(directory + '/*.data')
                for file in files:
                    future = ec.submit(AudioRepository.load_one_processed_audio, file)
                    processed_audio_futures.append(future)
            results = [future.result() for future in as_completed(processed_audio_futures)]

        return results

    @staticmethod
    def load_one_processed_audio(file: str):
        with open(file, 'rb') as f:
            loaded_object: tuple[FeatureVector, FeatureRepresentation] = pickle.load(f)
            loaded_object[0].audio.playlist = Path(file).parent.name
            return loaded_object
