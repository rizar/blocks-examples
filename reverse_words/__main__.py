#!/usr/bin/env python
"""Learn to reverse the letters in each word in text

In this demo, a recurrent network equipped with an attention mechanism
learns to reverse each word (on a character-by-character basis) in
its input text. The default training data is the Google Billion Word corpus,
which you should download and put to the path indicated in your .fuelrc file.

"""
import logging
import argparse
import glob
from reverse_words.main import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        "Case study of learning to reverse words in a text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "sample", "beam_search"],
        help="The mode to run. In the `train` mode a model is trained."
             " In the `sample` and `beam_search` modes a trained model is "
             " to used reverse words in the input text.")
    parser.add_argument(
        "save_path", default="reverse_words.zip", nargs='?',
        help="The path to save the training process if the mode"
             " is `train` OR path to an `.npz` files with learned"
             " parameters if the mode is `test`.")
    parser.add_argument(
        "--num-batches", default=10000, type=int,
        help="Train on this many batches.")
    parser.add_argument(
        "--data-path",
        help="text file(s) to read for training, with bash-like expansion"
             " so wildchars can be used (e.g. data/*.txt)")
    args = parser.parse_args()
    if args.data_path:
        args.data_path = glob.glob(args.data_path)
    main(**vars(args))
