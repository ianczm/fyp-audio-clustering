from glob import glob
from pathlib import Path
import processPlaylist


# Please ensure you have the data/extracted folder and data/raw folder.
# The raw folder should contain individual playlist folders containing .mp3 files.


def main():
    raw_dir = '../data/raw/*'
    raw_paths = [Path(directory) for directory in glob(raw_dir)]
    extracted_paths = [Path(p.parent.parent, 'extracted-compressed', p.name) for p in raw_paths]
    for r, e in zip(raw_paths, extracted_paths):
        processPlaylist.main(str(r), str(e))


if __name__ == '__main__':
    main()
