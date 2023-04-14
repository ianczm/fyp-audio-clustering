from src.helpers.repositories import AudioRepository
from src.helpers.loaders import RawAudioHandler, AudioProcessor
from src.models import AudioData
from concurrent.futures import ProcessPoolExecutor

import argparse
import os
from pathlib import Path


def parse_args() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    return args['directory']


# Use environment variable DISABLE_CLI=1 to enable CLI input
def prompt_args():
    return input('Playlist directory: ')


def get_directory():
    if os.environ.get('DISABLE_CLI') == '1':
        return prompt_args()
    else:
        return parse_args()


def process(directory: str, audio_data: AudioData):
    print(f'Processing {audio_data.name}')
    processed = AudioProcessor(audio_data).process()
    print(f'Done {audio_data.name}')
    AudioRepository.store_processed_audio(directory, processed)
    print(f'Saved {audio_data.name}')


def process_directory(directory: str):
    return lambda audio_data: process(directory, audio_data)


def main():
    directory = get_directory()
    raw_audio_handler = RawAudioHandler(directory)
    raw_audio = raw_audio_handler.load()
    with ProcessPoolExecutor() as ec:
        ec.map(process_directory(directory), raw_audio)


if __name__ == '__main__':
    main()
