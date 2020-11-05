from unittest import TestCase

# from pp import regr_likel, PointProcessDataset, InterEventDistribution
# from pp.core.spectral import SpectralAnalyzer
from tests.data import SpectralData


class TestSpectral(TestCase):
    def setUp(self) -> None:
        self.data = SpectralData()

    def test_spectral_analyzer(self):
        pass
        # TODO for now the spectral analysis has not been tested yet since it has been copied exactly from the
        #  original MATLAB script
        # model = regr_likel(PointProcessDataset.load(self.data.diff, 8), InterEventDistribution.INVERSE_GAUSSIAN)
        # spectral = SpectralAnalyzer(model)
        # res = spectral.psd()
