import sys
sys.path.append("..")

from glob import glob
from pathlib import Path
import processPlaylist


# Please ensure you have the data/extracted folder and data/raw folder.
# The raw folder should contain individual playlist folders containing .mp3 files.
RAW_DIR = '../data/raw/*'


def string_to_playlist_indices(string: str):
    return [int(s.strip()) for s in string.split(',')]


def select_playlists():
    [print(f'{idx}: {Path(path).name}') for idx, path in enumerate(glob(RAW_DIR))]
    playlist_indices = string_to_playlist_indices(input('Select playlists (comma-separated indices): '))
    playlists = [path for idx, path in enumerate(glob(RAW_DIR)) if idx in playlist_indices]
    print(f'You have selected: {playlists}')
    return playlists


def main():
    playlists = select_playlists()
    raw_paths = [Path(directory) for directory in playlists]
    extracted_paths = [Path(p.parent.parent, 'extracted', p.name) for p in raw_paths]
    for r, e in zip(raw_paths, extracted_paths):
        processPlaylist.main(str(r), str(e))


if __name__ == '__main__':
    main()
