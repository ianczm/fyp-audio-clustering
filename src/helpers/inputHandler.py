from glob import glob
import numpy as np
import pandas as pd

from src.models.features import *
from src.helpers.constants import CHORD_MAP, MIDI_MAX_NOTE

import librosa
import librosa.display
import librosa.feature
import librosa.onset
from madmom.features import notes, key, chords
from sklearn.decomposition import PCA

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool


class RawAudioHandler:
    audio_files: list[str]
    data: list[AudioData]

    def __init__(self, path_to_directory: str, limit: int = 0):
        audio_files = glob(path_to_directory + '/*.mp3')
        if limit > 0:
            self.audio_files = audio_files[:limit]
        else:
            self.audio_files = audio_files

    def load(self):
        # with ProcessPoolExecutor() as ec:
        #     self.data = list(ec.map(self.__load_one, self.audio_files))
        # with Pool() as pool:
        #     self.data = pool.map(self.__load_one, self.audio_files)
        self.data = list(map(self.__load_one, self.audio_files))
        return self.data

    def __load_one(self, audio_file: str):
        waveform, sample_rate = librosa.load(audio_file)
        return AudioData(waveform=waveform, sample_rate=sample_rate)


class AudioProcessor:
    # Processing Parameters
    n_mels: int
    n_mfcc: int
    hop_length: int
    frame_size: int
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
                 n_mels=256,
                 n_mfcc=13,
                 hop_length=512,
                 frame_size=2048,
                 chord_map=CHORD_MAP,
                 ignore_non_chords=True):
        self.feature_vector = FeatureVector(audio=audio)
        self.audio = audio
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
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
        self.__to_mfcc()
        # -- Temporal
        self.__to_zero_crossings()
        # -- Harmonic
        self.__to_key_signature()
        self.__to_note_trajectory()

        # Must be executed in order
        self.__to_bpm()                                             # 1
        self.__to_chroma_cqt(self.beat_frames)                      # 2
        self.__to_tonnetz(self.feature_repr.chroma_cqt_sync)        # 3.1
        self.__to_chord_trajectory(self.feature_repr.chroma_cqt)    # 3.2

        return self.feature_vector, self.feature_repr

    def __to_spectrogram(self):
        stft_data = librosa.stft(self.audio.waveform)
        spectrogram = librosa.amplitude_to_db(np.abs(stft_data), ref=np.max)

        self.feature_repr.spectrogram = spectrogram

    def __to_mel_spectrogram(self):
        mel_data = librosa.feature.melspectrogram(y=self.audio.waveform, sr=self.audio.sample_rate, n_mels=self.n_mels)
        mel_spectrogram = librosa.amplitude_to_db(np.abs(mel_data), ref=np.max)

        self.feature_repr.mel_spectrogram = mel_spectrogram

    def __to_spectral_centroid(self):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio.waveform, sr=self.audio.sample_rate)

        self.feature_vector.spectral.spectral_centroid_mean = spectral_centroid.mean()
        self.feature_vector.spectral.spectral_centroid_var = spectral_centroid.var()

        self.feature_repr.spectral_centroid = spectral_centroid[0]

    def __to_spectral_rolloff(self):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=self.audio.waveform, sr=self.audio.sample_rate)

        self.feature_vector.spectral.spectral_rolloff_mean = spectral_rolloff.mean()
        self.feature_vector.spectral.spectral_rolloff_var = spectral_rolloff.var()

        self.feature_repr.spectral_rolloff = spectral_rolloff[0]

    def __to_spectral_flux(self):
        # Spectral flux
        # squared distance between normalised magnitudes of successive spectral distributions
        spectral_flux = librosa.onset.onset_strength(y=self.audio.waveform, sr=self.audio.sample_rate)

        self.feature_vector.spectral.spectral_flux_mean = spectral_flux.mean()
        self.feature_vector.spectral.spectral_flux_var = spectral_flux.var()

        self.feature_repr.spectral_flux = spectral_flux

    def __to_mfcc(self):
        # 13 MFCC coefficients, and using only the first 5 excluding DC component
        cepstral_coefficients = librosa.feature.mfcc(y=self.audio.waveform, sr=self.audio.sample_rate, n_mfcc=self.n_mfcc)

        self.feature_vector.spectral.mfcc_mean_1 = cepstral_coefficients[1].mean()
        self.feature_vector.spectral.mfcc_mean_2 = cepstral_coefficients[2].mean()
        self.feature_vector.spectral.mfcc_mean_3 = cepstral_coefficients[3].mean()
        self.feature_vector.spectral.mfcc_mean_4 = cepstral_coefficients[4].mean()
        self.feature_vector.spectral.mfcc_mean_5 = cepstral_coefficients[5].mean()
        self.feature_vector.spectral.mfcc_var_1 = cepstral_coefficients[1].var()
        self.feature_vector.spectral.mfcc_var_2 = cepstral_coefficients[2].var()
        self.feature_vector.spectral.mfcc_var_3 = cepstral_coefficients[3].var()
        self.feature_vector.spectral.mfcc_var_4 = cepstral_coefficients[4].var()
        self.feature_vector.spectral.mfcc_var_5 = cepstral_coefficients[5].var()

        self.feature_repr.mfccs = cepstral_coefficients[1:6]

    def __to_zero_crossings(self):
        zero_crossings = librosa.zero_crossings(y=self.audio.waveform)

        self.feature_vector.temporal.zero_crossings_mean = zero_crossings.mean()
        self.feature_vector.temporal.zero_crossings_var = zero_crossings.var()

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

        for x in range(0, chord_beat_matrix.shape[0]-1):
            chord_x = chord_beat_matrix['chord'].iloc[x]
            chord_y = chord_beat_matrix['chord'].iloc[x+1]

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
        return chord_trajectory

    def __to_note_trajectory(self):
        note_peak_proc = notes.NotePeakPickingProcessor(fps=self.audio.sample_rate/self.hop_length)
        piano_note_proc = notes.RNNPianoNoteProcessor()(self.audio.waveform)
        note_time_matrix = note_peak_proc(piano_note_proc)

        note_trajectory = self.__construct_note_trajectory(note_time_matrix)
        self.feature_repr.note_trajectory = note_trajectory

        note_vector = self.__process_note_trajectory_as_feature(note_trajectory)
        self.feature_vector.harmonic.note_trajectory = note_vector

    def __construct_note_trajectory(self, note_time_matrix: ndarray):
        note_trajectory = np.zeros(shape=(MIDI_MAX_NOTE, MIDI_MAX_NOTE))

        for x in range(note_time_matrix.shape[0]-1):
            note_x = int(note_time_matrix[x][1])
            note_y = int(note_time_matrix[x+1][1])
            note_trajectory[note_y, note_x] += 1

        return note_trajectory

    # Todo: implement (dimensionality reduction)
    def __process_note_trajectory_as_feature(self, note_trajectory: ndarray):
        return note_trajectory









