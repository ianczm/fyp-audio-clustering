# Extracted Data

Extracted Data are the feature vectors extracted from the raw `.mp3` audio files.

They are stored here for persistence so to avoid having to recompute features on each run. Instead, the clustering algorithm simply reads the feature vectors from here.

## Filesystem

Place and categorise your extracted playlists here.

- Each playlist is a folder.
- Folder names will be treated as playlist names.
- Only `.data` files will be read.
- Does not have to follow the format in the `data/raw` folder.

## Converting Raw `.mp3` to `.data`

- Run the `processPlaylist.py` script under `src/scripts`.

[//]: # (Todo: processAllPlaylists should be able to process the entire raw folder)
