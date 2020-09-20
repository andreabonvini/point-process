import argparse

from pp.algorithmic import regr_likel
from pp.utils import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-path",
        type=str,
        help="path to .csv dataset",
        default="../tests/test_data/Y2.csv",
    )
    args = parser.parse_args()
    events = load(args.path)
    events = events[269:299]
    thetap, kappa, opt = regr_likel(events)
