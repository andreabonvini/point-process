import matplotlib.pyplot as plt
import numpy as np


def ppplot(
    times: np.ndarray, mus: np.ndarray, events: np.ndarray, targets: np.ndarray
):  # pragma: no cover
    target_bins = [t for t in targets if t]
    target_times = [times[i] for i in range(len(times)) if events[i]]
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(times, mus, label="First moment of IG regression", color="blue")
    ax.plot(target_times, target_bins, "*", label="RR", color="red")
    ax.legend(loc="best", fontsize="x-large")
    plt.xlabel("Time [s]")
    plt.ylabel("Inter Event Time [s]")
    plt.show()


def lambda_plot(times: np.ndarray, lambdas: np.ndarray):  # pragma: no cover
    plt.figure(figsize=(10, 5))
    plt.plot(times, lambdas)
    plt.xlabel("Time [s]")
    plt.legend(["Hazard Rate Function λ"])
    plt.show()
