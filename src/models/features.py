from numpy import ndarray
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(kw_only=True)
class AbstractFeature:
    def as_dict(self):
        return asdict(self)


@dataclass(kw_only=True)
class AudioData(AbstractFeature):
    name: str
    playlist: str = 'None'
    # given by librosa.load()
    waveform: Optional[ndarray] = None
    sample_rate: Optional[float] = None

    def as_dict(self):
        return {
            'song_name': self.get_song_name(),
            'artist': self.get_artists()[0],
            'playlist': self.playlist
        }

    def get_artists(self):
        # name: Artist1, Artist2 - Title
        return self.name.partition(' - ')[0].split(', ')

    def get_song_name(self):
        return self.name.partition(' - ')[2]


@dataclass(kw_only=True)
class TemporalFeatures(AbstractFeature):
    # 3 features
    zero_crossings_mean: Optional[float] = None
    zero_crossings_var: Optional[float] = None
    bpm: Optional[float] = None


@dataclass(kw_only=True)
class SpectralFeatures(AbstractFeature):
    # 16 features (missing the low-energy feature)
    spectral_centroid_mean: Optional[float] = None
    spectral_centroid_var: Optional[float] = None
    spectral_rolloff_mean: Optional[float] = None
    spectral_rolloff_var: Optional[float] = None
    spectral_flux_mean: Optional[float] = None
    spectral_flux_var: Optional[float] = None
    mfcc_mean_1: Optional[float] = None
    mfcc_var_1: Optional[float] = None
    mfcc_mean_2: Optional[float] = None
    mfcc_var_2: Optional[float] = None
    mfcc_mean_3: Optional[float] = None
    mfcc_var_3: Optional[float] = None
    mfcc_mean_4: Optional[float] = None
    mfcc_var_4: Optional[float] = None
    mfcc_mean_5: Optional[float] = None
    mfcc_var_5: Optional[float] = None


@dataclass(kw_only=True)
class HarmonicFeatures(AbstractFeature):
    # 2 arrays based on clustering paper
    chord_trajectory: Optional[ndarray] = None
    note_trajectory: Optional[ndarray] = None
    key_signature: Optional[int] = None  # pitch class


@dataclass(kw_only=True)
class FeatureVector(AbstractFeature):
    # Only used for processing
    audio: AudioData
    # Will be converted to DataFrame
    temporal: TemporalFeatures
    spectral: SpectralFeatures
    harmonic: HarmonicFeatures

    def as_dict(self):
        return self.audio.as_dict() \
            | self.temporal.as_dict() \
            | self.spectral.as_dict()
            # | self.harmonic.as_dict()


@dataclass(kw_only=True)
class FeatureRepresentation(AbstractFeature):
    spectrogram: Optional[ndarray] = None
    mel_spectrogram: Optional[ndarray] = None
    chroma_cqt: Optional[ndarray] = None
    chroma_cqt_sync: Optional[ndarray] = None
    tonnetz: Optional[ndarray] = None
    spectral_centroid: Optional[ndarray] = None
    spectral_rolloff: Optional[ndarray] = None
    spectral_flux: Optional[ndarray] = None
    zero_crossings: Optional[ndarray] = None
    mfccs: Optional[ndarray] = None
    chord_trajectory: Optional[ndarray] = None
    note_trajectory: Optional[ndarray] = None
