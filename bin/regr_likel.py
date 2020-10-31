import argparse
import os

# noinspection PyUnresolvedReferences
import fix_path  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pp import InterEventDistribution
from pp.core.distributions import events2interevents
from pp.core.model import PointProcessDataset
from pp.regression import regr_likel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path", "-p", type=str, help="path to .csv dataset",
    )
    parser.add_argument(
        "--ar_order", "-a", type=int, help="AR order for the mean parameter", default=9
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="output directory for plots / results etc...",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    data = np.array(pd.read_csv(args.path))
    n_samples = 150
    fs = data[0, 6]
    events = data[:n_samples, 0] / fs
    model = regr_likel(events, InterEventDistribution.INVERSE_GAUSSIAN, args.ar_order)

    # plot some result
    diffs = events2interevents(events)
    k_history = []
    theta_history = []
    for params in model.params_history:
        k_history.append(params[0])
        theta_history.append(params[1:])

    plt.figure(num=None, figsize=(15, 10), dpi=120, facecolor="w", edgecolor="k")
    plt.plot(diffs)
    plt.savefig(f"{args.output_dir}/data.png")
    plt.clf()

    dataset = PointProcessDataset.load(diffs, args.ar_order, True)
    test_data = dataset.xn[:, 1:]
    targets = dataset.wn.reshape(dataset.wn.shape[0],)
    mu_predictions = np.array([model(sample).mu for sample in test_data])
    residuals = mu_predictions - targets

    plt.figure(num=None, figsize=(15, 10), dpi=120, facecolor="w", edgecolor="k")
    plt.plot(range(len(mu_predictions)), mu_predictions, "r")
    plt.plot(range(len(targets)), targets, "b")

    plt.legend(labels=["predictions", "targets"])
    plt.savefig(f"{args.output_dir}/plot.png")
    plt.clf()

    plt.figure(num=None, figsize=(15, 10), dpi=120, facecolor="w", edgecolor="k")
    for p_hist in np.array(theta_history).T:
        plt.plot(range(len(p_hist)), p_hist)
    plt.legend(labels=[f"theta_{str(i)}" for i in range(len(theta_history[0]))])
    plt.savefig(f"{args.output_dir}/theta_history.png")
    plt.clf()

    plt.figure(num=None, figsize=(15, 10), dpi=120, facecolor="w", edgecolor="k")
    plt.plot(range(len(k_history)), k_history, "r")
    plt.legend(labels=["k history"])
    plt.savefig(f"{args.output_dir}/k_history.png")
    plt.clf()

    plt.figure(num=None, figsize=(15, 10), dpi=120, facecolor="w", edgecolor="k")
    plt.plot(range(len(model.results)), model.results, "r")
    plt.legend(labels=["negloglikelihood"])
    plt.savefig(f"{args.output_dir}/optimization_values.png")
    plt.clf()

    mse = np.mean(np.sqrt(residuals ** 2))

    print(f"MSE: {mse}")
