import argparse
import logging

from markov_chain import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        "Case study of generating a Markov chain with RNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "sample"],
        help="The mode to run. Use `train` to train a new model"
             " and `sample` to sample a sequence generated by an"
             " existing one.")
    parser.add_argument(
        "save_path", default="chain",
        help="The path to save the training process.")
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of steps to samples.")
    parser.add_argument(
        "--num-batches", default=1000, type=int,
        help="Train on this many batches.")
    args = parser.parse_args()
    main(**vars(args))
