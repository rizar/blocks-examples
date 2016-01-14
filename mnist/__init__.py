#!/usr/bin/env python

import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import MLP, Tanh, Softmax, WEIGHT
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from blocks.log import TrainingLog
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks_extras.extensions.synchronization import (
    Synchronize, SynchronizeWorker)
from platoon.param_sync import EASGD, ASGD

try:
    from blocks.extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


def main(save_to, num_epochs,
         learning_rate, sync_freq, rule, alpha):
    mlp = MLP([Tanh(), Softmax()], [784, 500, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    mlp.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    probs = mlp.apply(x)
    cost = CategoricalCrossEntropy().apply(y.flatten(), probs)
    error_rate = MisclassificationRate().apply(y.flatten(), probs)

    cg = ComputationGraph([cost])
    W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
    cost = cost + .00005 * (W1 ** 2).sum() + .00005 * (W2 ** 2).sum()
    cost.name = 'final_cost'

    mnist_train = MNIST(("train",))
    mnist_test = MNIST(("test",))

    worker = None
    if rule:
        worker = SynchronizeWorker(cport=1111, socket_timeout=2000)

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=learning_rate))
    extensions = [Timing()]
    if worker:
        if rule == 'asgd':
            sync_rule = ASGD()
        elif rule == 'easgd':
            sync_rule = EASGD()
        extensions += [
            Synchronize(worker, 'MNIST', sync_rule,
                        every_n_batches=sync_freq)]
    extensions += [
        FinishAfter(after_n_epochs=num_epochs),
        TrainingDataMonitoring(
            [cost, error_rate,
                aggregation.mean(algorithm.total_gradient_norm)],
            prefix="train",
            after_epoch=True)]
    if not worker or worker.is_main_worker:
        extensions += [
            DataStreamMonitoring(
                [cost, error_rate],
                Flatten(
                    DataStream.default_stream(
                        mnist_test,
                        iteration_scheme=ShuffledScheme(
                            mnist_test.num_examples, 500)),
                    which_sources=('features',)),
                    prefix="test"),
        Checkpoint(save_to)]
    extensions += [Printing()]

    if BLOCKS_EXTRAS_AVAILABLE:
        extensions.append(Plot(
            'MNIST example',
            channels=[
                ['test_final_cost',
                 'test_misclassificationrate_apply_error_rate'],
                ['train_total_gradient_norm']]))

    log = TrainingLog()
    # This will show up in the output of the training process
    if worker:
        log.status['is_main_worker'] = worker.is_main_worker
    main_loop = MainLoop(
        algorithm,
        Flatten(
            DataStream.default_stream(
                mnist_train,
                iteration_scheme=ShuffledScheme(
                    mnist_train.num_examples, 50,
                    rng=numpy.random.RandomState(
                        worker.seed if worker else 1))),
            which_sources=('features',)),
        model=Model(cost),
        log=log,
        extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs to do.")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="Learning rate for SGD")
    parser.add_argument("--sync-freq", type=int, default=10,
                        help="Synchronization frequency")
    parser.add_argument("--rule", choices=[None, 'asgd', 'easgd'],
                        help="Synchronization frequency")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Alpha parameter for EASGD. If given"
                             " training process is synchronized with"
                             " similar instances.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs,
         args.learning_rate, args.sync_freq, args.rule, args.alpha)
