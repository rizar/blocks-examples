#!/usr/bin/env python

import logging
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import MLP, Tanh, Softmax, WEIGHT
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop
from blocks.bricks.lookup import LookupTable
from blocks.theano_expressions import l2_norm

try:
    from blocks.extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


def main(save_to, num_epochs):
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    batch_size = x.shape[0]

    mlp_student = MLP([Tanh(), Softmax()], [784, 100, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0), name='mlp_student')
    mlp_student.initialize()
    mlp_teacher = MLP([Tanh(), Softmax()], [784, 100, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0), name='mlp_teacher')
    mlp_teacher.initialize()

    lookup = LookupTable(10, 784, weights_init=IsotropicGaussian(0.01))
    lookup.initialize()
    x_teacher = x + lookup.apply(y.flatten())

    # Teacher tries to perform well
    probs_teacher = mlp_teacher.apply(tensor.flatten(x_teacher, outdim=2))
    cost_teacher = CategoricalCrossEntropy().apply(y.flatten(), probs_teacher)
    cost_teacher.name = 'cost_teacher'
    error_rate_teacher = MisclassificationRate().apply(y.flatten(), probs_teacher)
    error_rate_teacher.name = 'error_rate_teacher'

    # Student tries to imitate teacher
    probs_student = mlp_student.apply(tensor.flatten(x, outdim=2))
    error_rate_student = MisclassificationRate().apply(y.flatten(), probs_student)
    error_rate_student.name = 'error_rate_student'

    hiddens_teacher = VariableFilter(bricks=[Tanh])(ComputationGraph(probs_teacher))
    hiddens_student = VariableFilter(bricks=[Tanh])(ComputationGraph(probs_student))
    discrepancy = l2_norm(
        [h_student - h_teacher
         for h_student, h_teacher in zip(hiddens_student, hiddens_teacher)])
    discrepancy = discrepancy ** 2 / 100 / batch_size
    discrepancy += tensor.nnet.categorical_crossentropy(probs_student, probs_teacher).mean()
    discrepancy.name = 'discrepancy'

    cost = cost_teacher + discrepancy
    cost.name = 'cost'

    cg = ComputationGraph([cost])

    mnist_train = MNIST(("train",))
    mnist_test = MNIST(("test",))

    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=Scale(learning_rate=0.1))
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost_teacher, cost, error_rate_teacher, error_rate_student],
                      Flatten(
                          DataStream.default_stream(
                              mnist_test,
                              iteration_scheme=SequentialScheme(
                                  mnist_test.num_examples, 500)),
                          which_sources=('features',)),
                      prefix="test"),
                  TrainingDataMonitoring(
                      [cost_teacher, cost, discrepancy, error_rate_teacher, error_rate_student,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True,
                      every_n_batches=None),
                  Checkpoint(save_to),
                  Printing(every_n_batches=None)]

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs)
