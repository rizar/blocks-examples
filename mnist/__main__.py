import logging
from argparse import ArgumentParser

from mnist import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("--criterion", type=str, choices=['nll', 'pg'],
                        help="Training criterion to use")
    parser.add_argument("--scale", type=float,
                        help="Scale")
    parser.add_argument("--momentum", type=float,
                        help="Momentum")
    parser.add_argument("save_to", default="mnist", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(**args.__dict__)
