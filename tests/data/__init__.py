import numpy as np
import pandas as pd


class DatasetTestData:
    def __init__(self):
        self.events = np.array([300.0, 600.0, 1000.0, 1450.0, 1700.0, 2100.0, 2600.0])
        self.inter_events_times = np.array([300.0, 400.0, 450.0, 250.0, 400.0, 500.0])
        self.p = 3
        self.wn = np.array([[250.0], [400.0], [500.0]])
        self.xn = np.array(
            [
                [1.0, 450.0, 400.0, 300.0],
                [1.0, 250.0, 450.0, 400.0],
                [1.0, 400.0, 250.0, 450.0],
            ]
        )
        self.k = 1000.0
        self.theta = np.array([[10.0], [5.0], [-2.0], [1.0]])
        self.eta = np.array([[0.50], [0.75], [1.0]])
        self.mus = np.dot(self.xn, self.theta)
        self.params = np.vstack([[[self.k]], self.theta]).squeeze()


class SpectralData:
    def __init__(self):
        df = pd.read_csv("tests/data/test_dataset.csv")
        rr = np.array(df["RR"])
        self.rr = rr
        self.diff = np.diff(rr)
        self.expected_poles = []
