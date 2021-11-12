import argparse
import numpy as np

from analysis import sgd, FFNN, logistic


if __name__ == "__main__":
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="To run the analysis and the tests")
    parser.add_argument(
        "-s",
        "--sgd",
        help="To run the analysis for doing OLS and Ridge regression using the SGD method",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--ffnn",
        help="To run the analysis for the Feedforward neural network",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--logistic",
        help="To run the analysis for the logistic regression",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--all",
        help="To run all the analyzes",
        action="store_true",
    )
    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
    if args.sgd or args.all:
        sgd.main()
    if args.ffnn or args.all:
        FFNN.main()
    if args.logistic or args.all:
        logistic.main()
