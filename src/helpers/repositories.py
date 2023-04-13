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

    @staticmethod
    def load_processed_audio(directory: str, start: int = 0, limit: int = 0):
        processed_audio: list[tuple[FeatureVector, FeatureRepresentation]] = []
        if limit > 0:
            files = glob(directory + '/*.data')[start:start + limit]
        else:
            files = glob(directory + '/*.data')
        for file in files:
            with open(file, 'rb') as f:
                processed_audio.append(pickle.load(f))
        return processed_audio
