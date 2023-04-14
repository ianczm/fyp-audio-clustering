import argparse
import os
from concurrent.futures import ProcessPoolExecutor, wait

from src.helpers.processors import AudioDataProcessor, FeatureVectorProcessor
from src.helpers.repositories import AudioRepository
from src.models import AudioData


def parse_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_dir')
    parser.add_argument('extracted_dir')
    args = parser.parse_args()
    return args['raw_dir'], args['extracted_dir']


# Use environment variable DISABLE_CLI=1 to enable CLI input
def prompt_args():
    raw = input('Raw playlist directory: ')
    extracted = input('Extracted playlist directory: ')
    return raw, extracted


def get_directory():
    if os.environ.get('DISABLE_CLI') == '1':
        return prompt_args()
    else:
        return parse_args()


def process(directory: str, audio_data: AudioData):
    print(f'Processing {audio_data.name}')
    processed = FeatureVectorProcessor(audio_data).process()
    print(f'Done {audio_data.name}')
    AudioRepository.store_one_processed_audio(directory, processed)
    print(f'Saved {audio_data.name}')


def main():
    raw_directory, extracted_directory = get_directory()
    raw_audio_handler = AudioDataProcessor(raw_directory)
    raw_audio = raw_audio_handler.load()
    with ProcessPoolExecutor() as ec:
        futures = [ec.submit(process, extracted_directory, audio_data) for audio_data in raw_audio]
        wait(futures)


if __name__ == '__main__':
    main()
