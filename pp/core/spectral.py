from copy import deepcopy  # pragma: no cover
from typing import List, NamedTuple  # pragma: no cover

import numpy as np  # pragma: no cover

from pp.core.model import PointProcessModel  # pragma: no cover


class Pole(NamedTuple):  # pragma: no cover
    # TODO add Documentation
    pos: complex  # position on the complex plane
    frequency: float
    power: float
    residual: float
    comps: List[complex]  # spectral components for the specific pole


class SpectralAnalysis(NamedTuple):  # pragma: no cover
    frequencies: np.array  # Hz
    powers: np.array  # mm^2 / Hz
    poles: List[Pole]


class SpectralAnalyzer:  # pragma: no cover
    def __init__(self, model: PointProcessModel):  # pragma: no cover
        self.model = model

    def psd(self) -> SpectralAnalysis:  # pragma: no cover
        """
        Compute Power Spectral Density for the given model.
        """
        return self._compute_psd(self.model.theta, self.model.wn, self.model.k)

    def _compute_psd(
        self, theta: np.ndarray, wn: np.ndarray, k: float
    ) -> SpectralAnalysis:  # pragma: no cover
        thetap = deepcopy(theta[1:]) if self.model.hasTheta0 else deepcopy(theta[:])
        mean_interval = np.mean(wn)
        var = mean_interval ** 3 / k
        var = 1e6 * var  # from [s^2] to [ms^2]
        fsamp = 1 / mean_interval
        ar = np.r_[1, -thetap]  # [1, -θ1, -θ2, ..., θp]
        # Compute poles' complex values
        poles_values = np.roots(ar)
        # Order them by absolute angle
        poles_values = sorted(poles_values, key=lambda x: abs(np.angle(x)))

        # Fix AR models that might have become slightly unstable due to the estimation process
        # using an exponential decay (see Stoica and Moses, Signal Processing 26(1) 1992)
        mod_scale = min(0.99 / max(np.abs(poles_values)), 1)
        poles_values = mod_scale * poles_values
        thetap = thetap * np.cumprod(np.ones(thetap.shape) * mod_scale)

        nf = 1024
        fs = np.linspace(0, 0.5, nf)  # normalized freq
        # z = e^(-2πfT)
        # z: unit delay operator
        z = np.exp(2j * np.pi * fs)
        # P(z) = (σ^2*T)/ |1+θ1*z^(-1)+...+θp*z^(-p)|^2
        # σ^2 : Sample variance
        # T: sampling interval
        powers = (var / fsamp) / abs(np.polyval(np.r_[1, -thetap], np.conj(z))) ** 2
        frequencies = fs * fsamp
        poles_residuals = [
            1
            / (
                p
                * np.prod(p - [val for val in poles_values if val is not p])
                * np.prod(1 / p - np.conj(poles_values))
            )
            for p in poles_values
        ]

        poles_frequencies = [np.angle(p) / (2 * np.pi) * fsamp for p in poles_values]
        poles_powers = [var * np.real(p) for p in poles_residuals]
        # We also save the spectral components for each frequency value for each pole
        poles_comps = []
        ref_poles = 1 / np.conj(poles_values)
        for i in range(len(poles_values)):
            pp = poles_residuals[i] * poles_values[i] / (z - poles_values[i])
            refpp = -np.conj(poles_residuals[i]) * ref_poles[i] / (z - ref_poles[i])
            poles_comps.append(var / fsamp * (pp + refpp))

        poles = [
            Pole(pos, freq, power, res, comps)
            for pos, freq, power, res, comps in zip(
                poles_values,
                poles_frequencies,
                poles_powers,
                poles_residuals,
                poles_comps,
            )
        ]

        return SpectralAnalysis(frequencies, powers, poles)
