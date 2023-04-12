from src.helpers.repositories import AudioRepository
from src.helpers.loaders import RawAudioHandler, AudioProcessor
from src.models import AudioData
from concurrent.futures import ProcessPoolExecutor


DIRECTORY = '../../data/playlist-1'


def process(audio_data: AudioData):
    print(f'Processing {audio_data.name}')
    processed = [AudioProcessor(audio_data).process()]
    print(f'Done {audio_data.name}')
    AudioRepository.store_processed_audio(DIRECTORY, processed)
    print(f'Saved {audio_data.name}')


def main():
    raw_audio_handler = RawAudioHandler(DIRECTORY)
    raw_audio = raw_audio_handler.load()
    with ProcessPoolExecutor() as ec:
        ec.map(process, raw_audio)


if __name__ == '__main__':
    main()
