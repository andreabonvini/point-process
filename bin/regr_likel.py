import argparse

# noinspection PyUnresolvedReferences
import fix_path  # noqa: F401

from pp.core.utils import load
from pp.regression import regr_likel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-path",
        type=str,
        help="path to .csv dataset",
        default="tests/test_data/Y2.csv",
    )
    args = parser.parse_args()
    events = load(args.path)[70:120]
    res = regr_likel(events)
    breakpoint()
