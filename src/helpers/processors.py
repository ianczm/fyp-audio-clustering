from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import librosa
import librosa.display
import librosa.feature
import librosa.onset
import numpy as np
import pandas as pd
from madmom.features import notes, key, chords
from scipy.ndimage import zoom

from src.helpers.constants import CHORD_MAP, MIDI_MAX_NOTE, TONNETZ_LENGTH
from src.models.features import *


# Todo: Refactor to use Path exclusively
class AudioDataProcessor:
    audio_files: list[str]
    data: list[AudioData]
    path_to_directory: str
    limit: int

    def __init__(self, path_to_directory: str, limit: int = 0):
        self.path_to_directory = path_to_directory
        self.limit = limit
        self.__read_filenames()

    def __read_filenames(self):
        audio_files = glob(self.path_to_directory + '/*.mp3')
        if self.limit > 0:
            self.audio_files = audio_files[:self.limit]
        else:
            self.audio_files = audio_files

    def load(self):
        with ProcessPoolExecutor() as ec:
            self.data = list(ec.map(AudioDataProcessor.load_one, self.audio_files))
        return self.data

    @staticmethod
    def load_one(audio_file: str, playlist: str = None):
        waveform, sample_rate = librosa.load(audio_file)
        name = Path(audio_file).stem
        audio_data = AudioData(name=name, waveform=waveform, sample_rate=sample_rate)
        if playlist is not None:
            audio_data.playlist = playlist
        return audio_data

    @staticmethod
    def load_spotify_song(url: str) -> AudioData:
        pass


class FeatureVectorProcessor:
    # Processing Parameters
    n_mels: int
    n_mfcc: int
    hop_length: int
    frame_size: int
    tonnetz_length: int
    chord_map: dict
    ignore_non_chords: bool

    # Storage Variables
    audio: AudioData
    feature_vector: FeatureVector
    feature_repr = FeatureRepresentation()

    # Intermediate Variables
    beat_frames: ndarray
    beat_times: ndarray

    def __init__(self,
                 audio: AudioData,
                 save_repr=False,
                 n_mels=256,
                 n_mfcc=13,
                 hop_length=512,
                 frame_size=2048,
                 tonnetz_length=TONNETZ_LENGTH,
                 chord_map=CHORD_MAP,
                 ignore_non_chords=True):
        self.feature_vector = FeatureVector(
            audio=audio,
            spectral=SpectralFeatures(),
            temporal=TemporalFeatures(),
            harmonic=HarmonicFeatures()
        )
        self.audio = audio
        self.save_repr = save_repr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.tonnetz_length = tonnetz_length
        self.frame_size = frame_size
        self.chord_map = chord_map
        self.ignore_non_chords = ignore_non_chords

    def process(self):

        # Can be executed in parallel
        # -- Spectral
        self.__to_spectrogram()
        self.__to_mel_spectrogram()
        self.__to_spectral_centroid()
        self.__to_spectral_rolloff()
        self.__to_spectral_flux()
        self.__to_spectral_flatness()
        self.__to_mfcc()
        # -- Temporal
        self.__to_zero_crossings()
        # -- Harmonic
        self.__to_key_signature()
        self.__to_note_trajectory()

        # Must be executed in order
        self.__to_bpm()  # 1
        self.__to_chroma_cqt(self.beat_frames)  # 2
        self.__to_tonnetz(self.feature_repr.chroma_cqt_sync)  # 3.1
        self.__to_chord_trajectory(self.feature_repr.chroma_cqt)  # 3.2

        return self.feature_vector, self.feature_repr

    def __to_spectrogram(self):
        stft_data = librosa.stft(self.audio.waveform)
        spectrogram = librosa.amplitude_to_db(np.abs(stft_data), ref=np.max)

        if self.save_repr:
            self.feature_repr.spectrogram = spectrogram

    def __to_mel_spectrogram(self):
        mel_data = librosa.feature.melspectrogram(y=self.audio.waveform, sr=self.audio.sample_rate, n_mels=self.n_mels)
        mel_spectrogram = librosa.amplitude_to_db(np.abs(mel_data), ref=np.max)

        if self.save_repr:
            self.feature_repr.mel_spectrogram = mel_spectrogram

    def __to_spectral_centroid(self):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio.waveform, sr=self.audio.sample_rate)

        self.feature_vector.spectral.spectral_centroid_mean = spectral_centroid.mean()
        self.feature_vector.spectral.spectral_centroid_var = spectral_centroid.var()

        if self.save_repr:
            self.feature_repr.spectral_centroid = spectral_centroid[0]

    def __to_spectral_rolloff(self):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio.waveform, sr=self.audio.sample_rate)

        self.feature_vector.spectral.spectral_rolloff_mean = spectral_rolloff.mean()
        self.feature_vector.spectral.spectral_rolloff_var = spectral_rolloff.var()

        if self.save_repr:
            self.feature_repr.spectral_rolloff = spectral_rolloff[0]

    def __to_spectral_flux(self):
        # Spectral flux
        # squared distance between normalised magnitudes of successive spectral distributions
        spectral_flux = librosa.onset.onset_strength(y=self.audio.waveform, sr=self.audio.sample_rate)

        self.feature_vector.spectral.spectral_flux_mean = spectral_flux.mean()
        self.feature_vector.spectral.spectral_flux_var = spectral_flux.var()

        if self.save_repr:
            self.feature_repr.spectral_flux = spectral_flux

    def __to_spectral_flatness(self):
        # Spectral flatness
        # high flatness = white noise, low flatness = musical
        spectral_flatness = librosa.feature.spectral_flatness(y=self.audio.waveform)

        self.feature_vector.spectral.spectral_flatness_mean = spectral_flatness.mean()
        self.feature_vector.spectral.spectral_flatness_var = spectral_flatness.var()

        if self.save_repr:
            self.feature_repr.spectral_flatness = spectral_flatness[0]

    def __to_mfcc(self):
        max_mfcc = 10

        # 13 MFCC coefficients, and using only the first 5 excluding DC component
        cepstral_coefficients = librosa.feature.mfcc(y=self.audio.waveform, sr=self.audio.sample_rate,
                                                     n_mfcc=self.n_mfcc)

        for i in range(1, max_mfcc+1):
            setattr(self.feature_vector.spectral, f'mfcc_mean_{i}', cepstral_coefficients[i].mean())
            setattr(self.feature_vector.spectral, f'mfcc_var_{i}', cepstral_coefficients[i].var())

        if self.save_repr:
            self.feature_repr.mfccs = cepstral_coefficients[1:max_mfcc+1]

    def __to_zero_crossings(self):
        zero_crossings = librosa.zero_crossings(y=self.audio.waveform)

        self.feature_vector.temporal.zero_crossings_mean = zero_crossings.mean()
        self.feature_vector.temporal.zero_crossings_var = zero_crossings.var()

        if self.save_repr:
            self.feature_repr.zero_crossings = zero_crossings

    def __to_key_signature(self):
        key_proc = key.CNNKeyRecognitionProcessor()
        global_key_prob = key_proc(self.audio.waveform)
        self.feature_vector.harmonic.key_signature = global_key_prob.argmax()

    # Todo: must be synchronous, but ignore first
    def __to_bpm(self):
        bpm, beat_frames = librosa.beat.beat_track(y=self.audio.waveform, sr=self.audio.sample_rate)

        self.beat_frames = beat_frames
        self.beat_times = librosa.frames_to_time(beat_frames)

        self.feature_vector.temporal.bpm = bpm

    # Todo: must be synchronous, but ignore first
    def __to_chroma_cqt(self, beat_frames):
        # Order of calling: bpm, chroma_cqt, chroma_sync, tonnetz

        # CQT used for harmonic content over STFT for rhythmic content
        chroma_cqt = librosa.feature.chroma_cqt(y=self.audio.waveform, sr=self.audio.sample_rate)
        self.feature_repr.chroma_cqt = chroma_cqt

        # np.max to get most prominent notes (beat_frames must be populated)
        chroma_cqt_sync = librosa.util.sync(chroma_cqt, beat_frames, aggregate=np.max)
        self.feature_repr.chroma_cqt_sync = chroma_cqt_sync

    # Todo: must be synchronous, but ignore first
    # should be chroma_cqt_sync, but chroma_cqt works too
    def __to_tonnetz(self, chroma_cqt):
        tonnetz = librosa.feature.tonnetz(sr=self.audio.sample_rate, chroma=chroma_cqt)

        self.feature_vector.harmonic.tonnetz = standardize_tonnetz(tonnetz, self.tonnetz_length)

        if self.save_repr:
            self.feature_repr.tonnetz = tonnetz

    # Todo: must be synchronous, but ignore first
    # should be chroma_cqt
    def __to_chord_trajectory(self, chroma_cqt):
        # Get a list of chords and the start/end time of their occurrences
        decode = chords.DeepChromaChordRecognitionProcessor(fps=self.audio.sample_rate / self.hop_length)
        chord_time_matrix = decode(chroma_cqt.T)

        # Group the chords by beat instead of based on arbitrary start/end time
        chord_beat_df = self.__construct_chord_beat_df(chord_time_matrix)

        # Accumulate occurrences into a chord trajectory matrix
        chord_trajectory = self.__construct_chord_trajectory(chord_beat_df)
        if self.save_repr:
            self.feature_repr.chord_trajectory = chord_trajectory

        # Process the matrix into the feature vector
        chord_vector = self.__process_chord_trajectory_as_feature(chord_trajectory)
        self.feature_vector.harmonic.chord_trajectory = chord_vector

    def __construct_chord_beat_df(self, chord_time_matrix: ndarray):
        start_idx, end_idx, current_idx = 0, 0, 0
        chord_beat_matrix: list = self.beat_times.tolist()

        for start_time, end_time, chord_label in chord_time_matrix:

            for enum, beat_time in enumerate(self.beat_times[start_idx:]):
                # get current index and set as end index, exclusive
                current_idx = enum + start_idx
                if beat_time < end_time:
                    chord_beat_matrix[current_idx] = beat_time, self.chord_map[chord_label]
                else:
                    break

            start_idx = current_idx

        return pd.DataFrame(chord_beat_matrix, columns=['beat_time', 'chord'])

    def __construct_chord_trajectory(self, chord_beat_matrix: pd.DataFrame):
        chord_count = len(self.chord_map)
        max_chord = chord_count - 1
        chord_trajectory = np.zeros(shape=(chord_count, chord_count))

        for x in range(0, chord_beat_matrix.shape[0] - 1):
            chord_x = chord_beat_matrix['chord'].iloc[x]
            chord_y = chord_beat_matrix['chord'].iloc[x + 1]

            if self.ignore_non_chords:
                if chord_x == max_chord and chord_y == max_chord:
                    continue
                elif chord_x == max_chord and chord_y != max_chord:
                    chord_trajectory[chord_y, chord_y] += 1
                elif chord_x != max_chord and chord_y == max_chord:
                    chord_trajectory[chord_x, chord_x] += 1
                else:
                    chord_trajectory[chord_y, chord_x] += 1
            else:
                chord_trajectory[chord_y, chord_x] += 1

        return chord_trajectory

    # Todo: implement (dimensionality reduction)
    def __process_chord_trajectory_as_feature(self, chord_trajectory: ndarray):
        return chord_trajectory.flatten()

    def __to_note_trajectory(self):
        note_peak_proc = notes.NoteOnsetPeakPickingProcessor(
            fps=self.audio.sample_rate / self.hop_length, pitch_offset=21)
        piano_note_proc = notes.RNNPianoNoteProcessor()(self.audio.waveform)
        note_time_matrix = note_peak_proc(piano_note_proc)

        note_trajectory = self.__construct_note_trajectory(note_time_matrix)
        if self.save_repr:
            self.feature_repr.note_trajectory = note_trajectory

        note_vector = self.__process_note_trajectory_as_feature(note_trajectory)
        self.feature_vector.harmonic.note_trajectory = note_vector

    def __construct_note_trajectory(self, note_time_matrix: ndarray):
        note_trajectory = np.zeros(shape=(MIDI_MAX_NOTE, MIDI_MAX_NOTE))

        for x in range(note_time_matrix.shape[0] - 1):
            note_x = int(note_time_matrix[x][1])
            note_y = int(note_time_matrix[x + 1][1])
            note_trajectory[note_y, note_x] += 1

        return note_trajectory

    # Todo: implement (dimensionality reduction)
    def __process_note_trajectory_as_feature(self, note_trajectory: ndarray):
        return note_trajectory.flatten()


def standardize_tonnetz(tonnetz: ndarray, tonnetz_length: int = TONNETZ_LENGTH):
    flattened_tonnetz = tonnetz.flatten()
    if len(flattened_tonnetz) != tonnetz_length:
        zoom_factor = tonnetz_length / len(flattened_tonnetz)
        standardized_tonnetz = zoom(flattened_tonnetz, zoom_factor)
        return standardized_tonnetz
    else:
        return flattened_tonnetz
