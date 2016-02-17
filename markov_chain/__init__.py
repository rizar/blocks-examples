#!/usr/bin/env python
"""Learn a Markov chain with an RNN and sample from it."""
from __future__ import print_function

import logging
import pprint
import sys

import numpy
import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.lookup import LookupTable
from blocks_extras.bricks.sequence_generator2 import (
    SequenceGenerator, SoftmaxReadout, LookupFeedback)
from blocks.graph import ComputationGraph
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from blocks.algorithms import GradientDescent, Scale
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.serialization import load
from blocks.main_loop import MainLoop
from blocks.select import Selector

from .dataset import MarkovChainDataset

sys.setrecursionlimit(10000)
floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def main(mode, save_path, steps, num_batches):
    num_states = MarkovChainDataset.num_states

    if mode == "train":
        # Experiment configuration
        rng = numpy.random.RandomState(1)
        batch_size = 50
        seq_len = 100
        dim = 10

        # Build the bricks and initialize them
        recurrent = SimpleRecurrent(name="transition", dim=dim,
                                    activation=Tanh())
        generator = SequenceGenerator(
            recurrent,
            SoftmaxReadout(dim=num_states,
                           merged_states=recurrent.apply.states),
            LookupFeedback(lookup=LookupTable(num_states),
                           # The next line assumes, that "mask" is the last
                           # in apply.sequences
                           feedback_sequences=recurrent.apply.sequences[:-1]),
            weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
            name="generator")
        generator.push_initialization_config()
        recurrent.weights_init = Orthogonal()
        generator.initialize()

        # Give an idea of what's going on.
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in Selector(generator).get_parameters().items()],
                        width=120))
        logger.info("Markov chain entropy: {}".format(
            MarkovChainDataset.entropy))
        logger.info("Expected min error: {}".format(
            -MarkovChainDataset.entropy * seq_len))

        # Build the cost computation graph.
        x = tensor.lmatrix('data')
        cost = generator.costs(x).mean()
        cost.name = "sequence_log_likelihood"

        algorithm = GradientDescent(
            cost=cost,
            parameters=ComputationGraph(cost).parameters,
            step_rule=Scale(0.1))
        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=DataStream(
                MarkovChainDataset(rng, seq_len),
                iteration_scheme=ConstantScheme(batch_size)),
            model=Model(cost),
            extensions=[FinishAfter(after_n_batches=num_batches),
                        TrainingDataMonitoring([cost], prefix="this_step",
                                               after_batch=True),
                        TrainingDataMonitoring([cost], prefix="average",
                                               every_n_batches=100),
                        Checkpoint(save_path, every_n_batches=500),
                        Printing(every_n_batches=100)])
        main_loop.run()
    elif mode == "sample":
        main_loop = load(open(save_path, "rb"))
        generator = main_loop.model.get_top_bricks()[-1]

        sample = ComputationGraph(generator.generate(
            n_steps=steps, batch_size=1, iterate=True)).get_theano_function()
        outputs, states = [data.squeeze() for data in sample()]

        freqs = numpy.bincount(outputs).astype(floatX)
        freqs /= freqs.sum()
        print("Frequencies:\n {} vs {}".format(freqs,
                                               MarkovChainDataset.equilibrium))

        trans_freqs = numpy.zeros((num_states, num_states), dtype=floatX)
        for a, b in zip(outputs, outputs[1:]):
            trans_freqs[a, b] += 1
        trans_freqs /= trans_freqs.sum(axis=1)[:, None]
        print("Transition frequencies:\n{}\nvs\n{}".format(
            trans_freqs, MarkovChainDataset.trans_prob))
    else:
        assert False
