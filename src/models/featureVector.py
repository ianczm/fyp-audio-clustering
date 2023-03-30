from numpy import ndarray


class AbstractFeature:
    def to_dict(self):
        pass


class TimeDomainFeatures(AbstractFeature):
    def __init__(self):
        self.waveform: ndarray
        self.samplerate: float
        self.zeroCrossingsMean: float
        self.zeroCrossingsVar: float
        self.bpm: float

    def to_dict(self):
        return [vars(s) for s in self]


class TimbreFeatures(AbstractFeature):
    def __init__(self):
        self.spectralCentroidMean: float
        self.spectralCentroidVar: float
        self.spectralRollOffMean: float
        self.spectralRollOffVar: float
        self.spectralFluxMean: float
        self.spectralFluxVar: float
        self.mfccMean1: float
        self.mfccVar1: float
        self.mfccMean2: float
        self.mfccVar2: float
        self.mfccMean3: float
        self.mfccVar3: float
        self.mfccMean4: float
        self.mfccVar4: float
        self.mfccMean5: float
        self.mfccVar5: float

    def to_dict(self):
        return [vars(s) for s in self]


class FeatureVector(AbstractFeature):
    def __init__(self):
        self.timeDomain = TimeDomainFeatures()
        self.timbre = TimbreFeatures()

    def to_dict(self):
        return self.timeDomain.to_dict() \
            + self.timbre.to_dict()
