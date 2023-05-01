# Scripts

This folder contains scripts specifically for converting the `.mp3` files to
`FeatureVector`s and `FeatureRepresentation`s in bulk.

This program also depends on the
[`spotify-downloader` package in the same folder](/scripts/spotify-downloader/README.md)
that provides access to Spotify Web API to query playlists and `youtube-dl`. This
package allows the user to specify a Spotify playlist to download as `.mp3`s.

The `save_repr` variable can be toggled to tell the program whether to save
`FeatureRepresentations` to disk. May take up significant space at about
1.5 GB per playlist of 50 songs.

---

## Filesystem

Using these scripts require that `/data/raw` adheres to the following:

- Each playlist is a folder.

- Folder names will be treated as playlist names.

- Only `.mp3` files will be read.

- `.features` and `.representations` files will be output to `/data/extracted`

Please also see the [Raw Data](/data/raw/README.md) and
[Extracted Data](/data/extracted/README.md) folder for more information.

---


## Usage

### Converting Raw `.mp3` to Extracted Data

- Run the `processPlaylist.py` or the `processAllPlaylists.py` script under
the [`/scripts` folder](/scripts).
- It will convert `.mp3` files found in subfolders into `.features.pkl.pbz2` files.
