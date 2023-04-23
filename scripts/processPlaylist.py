import argparse
import os
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd

from src.helpers.processors import AudioDataProcessor, FeatureVectorProcessor
from src.helpers.repositories import AudioRepository, CompressIO, PandasAudioRepository
from src.models import AudioData, FeatureVector, FeatureRepresentation


def parse_args() -> tuple[str, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')
    parser.add_argument('extracted_dir')
    parser.add_argument('--store-repr', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args['raw_dir'], args['extracted_dir'], args['--store-repr']


# Use environment variable DISABLE_CLI=1 to enable CLI input
def prompt_args() -> tuple[str, str, bool]:
    raw = input('Raw playlist directory: ')
    extracted = input('Extracted playlist directory: ')
    store_repr = input('Store representations? y/n: ')
    return raw, extracted, store_repr.lower() == 'y'


def get_args():
    if os.environ.get('DISABLE_CLI') == '1':
        return prompt_args()
    else:
        return parse_args()


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process(directory: str, audio_data: AudioData):
    print(f'Processing {audio_data.name}')
    processed = FeatureVectorProcessor(audio_data).process()
    print(f'Done {audio_data.name}')
    AudioRepository.store_one_processed_audio(directory, processed)
    print(f'Saved {audio_data.name}')


def process_without_saving(audio_data: AudioData, save_repr: bool):
    print(f'Processing {audio_data.name}')
    processed_audio = FeatureVectorProcessor(audio_data, save_repr).process()
    print(f'Done Processing {audio_data.name}')
    return processed_audio


def load_raw_audio(raw_directory: str, limit: int = 0):
    print(f'Loading audio...')
    raw_audio_handler = AudioDataProcessor(raw_directory, limit=limit)
    raw_audio = raw_audio_handler.load()
    return raw_audio


def save_vectors_as_dataframe(vectors: list[FeatureVector], extracted_directory: str):
    filename = Path(extracted_directory).name
    print(f'Saving dataframe for {filename}')

    def set_playlist_name_and_to_dict(vector: FeatureVector):
        vector.audio.playlist = filename
        return vector.as_dict()

    dataset = pd.DataFrame([set_playlist_name_and_to_dict(vector) for vector in vectors])
    dataset.to_pickle(extracted_directory + f'/{filename}.vectors.pkl.bz2', compression='bz2')
    print(f'Saved dataframe for {filename}')


def save_repr_as_pickle(feature_repr: FeatureRepresentation, directory: str):
    print(f'Saving repr as pickle for {Path(directory).name}')
    CompressIO.save(directory, feature_repr)
    print(f'Saved repr as pickle for {Path(directory).name}')


def main(raw_directory: str, extracted_directory: str, save_repr: bool = False):
    print(raw_directory, extracted_directory, save_repr)
    raw_audio = load_raw_audio(raw_directory)
    create_directory(extracted_directory)
    with ProcessPoolExecutor() as ec:
        processed_audio = list(ec.map(
            process_without_saving,
            raw_audio,
            [save_repr for _ in range(len(raw_audio))]
        ))
    print(f'Saving datasets...')
    if save_repr:
        PandasAudioRepository.store_datasets(
            extracted_directory,
            processed_audio
        )
    else:
        PandasAudioRepository.store_feature_dataset(
            extracted_directory,
            Path(extracted_directory).name,
            [v for v, _ in processed_audio]
        )
    print(f'Saved datasets')


def main_old(raw_directory: str, extracted_directory: str, save_repr: bool):
    raw_audio = load_raw_audio(raw_directory)
    create_directory(extracted_directory)
    with ProcessPoolExecutor() as ec:
        futures = [ec.submit(process, extracted_directory, audio_data) for audio_data in raw_audio]
        wait(futures)


if __name__ == '__main__':
    main(*get_args())
