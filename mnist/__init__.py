#!/usr/bin/env python

import logging
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Momentum
from blocks.bricks import MLP, Softmax, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

try:
    from blocks_extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


def main(save_to, scale, momentum, num_epochs):
    mlp = MLP([Rectifier(), Rectifier(), Softmax()], [784, 1000, 1000, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    mlp.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    probs = mlp.apply(x)
    cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
    error_rate = MisclassificationRate().apply(y.flatten(), probs)
    cg = ComputationGraph(cost)
    cost.name = 'final_cost'

    mnist_train = MNIST(("train",))
    mnist_test = MNIST(("test",))

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Momentum(learning_rate=scale, momentum=momentum))
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      Flatten(
                          DataStream.default_stream(
                              mnist_test,
                              iteration_scheme=SequentialScheme(
                                  mnist_test.num_examples, 500)),
                          which_sources=('features',)),
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  Printing()]

    if BLOCKS_EXTRAS_AVAILABLE:
        extensions.append(Plot(
            'MNIST example',
            channels=[
                ['test_final_cost',
                 'test_misclassificationrate_apply_error_rate'],
                ['train_total_gradient_norm']]))

    main_loop = MainLoop(
        algorithm,
        Flatten(
            DataStream.default_stream(
                mnist_train,
                iteration_scheme=SequentialScheme(
                    mnist_train.num_examples, 50)),
            which_sources=('features',)),
        model=Model(cost),
        extensions=extensions)

    main_loop.run()
