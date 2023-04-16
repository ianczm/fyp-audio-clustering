import os
import pickle
import bz2file as bz2
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from glob import glob
from pathlib import Path

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
        return '.data.pbz2'


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
        return '.data'


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
        loaded_object[0].audio.playlist = Path(file).parent.name
        return loaded_object

    # dir_start: which index to start from in each directory
    # dir_limit: how many songs to extract per directory
    @staticmethod
    def load_processed_audio(directories: list[str], dir_start: int = 0, dir_limit: int = 0):
        processed_audio_futures: list[Future[tuple[FeatureVector, FeatureRepresentation]]] = []

        with ProcessPoolExecutor() as ec:
            for directory in directories:
                if dir_limit > 0:
                    files = glob(directory + '/*' + AudioRepository.io.file_ext())[dir_start:dir_start + dir_limit]
                else:
                    files = glob(directory + '/*' + AudioRepository.io.file_ext())
                for file in files:
                    future = ec.submit(AudioRepository.load_one_processed_audio, file)
                    processed_audio_futures.append(future)
            results = [future.result() for future in as_completed(processed_audio_futures)]

        return results
