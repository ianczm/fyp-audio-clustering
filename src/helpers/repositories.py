import pickle
import os
from glob import glob
from src.models import *


EXTRACTED_SUFFIX = '-extracted'


class AudioRepository:
    @staticmethod
    def store_processed_audio(directory: str, processed_audio: list[tuple[FeatureVector, FeatureRepresentation]]):
        for audio in processed_audio:
            vector, _ = audio
            directory = directory + EXTRACTED_SUFFIX
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(f'{directory}/{vector.audio.name}.data', 'wb') as f:
                pickle.dump(audio, f)

    # dir_start: which index to start from in each directory
    # dir_limit: how many songs to extract per directory
    @staticmethod
    def load_processed_audio(directories: list[str], dir_start: int = 0, dir_limit: int = 0):
        processed_audio: list[tuple[FeatureVector, FeatureRepresentation]] = []

        for directory in directories:
            if dir_limit > 0:
                files = glob(directory + '/*.data')[dir_start:dir_start + dir_limit]
            else:
                files = glob(directory + '/*.data')
            for file in files:
                with open(file, 'rb') as f:
                    loaded_object: tuple[FeatureVector, FeatureRepresentation] = pickle.load(f)
                    loaded_object[0].audio.playlist = directory.split(os.path.sep)[-1]
                    processed_audio.append(loaded_object)

        return processed_audio
