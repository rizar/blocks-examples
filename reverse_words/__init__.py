from __future__ import print_function
import logging
import pprint
import math
import numpy
import traceback
import operator

import theano
from six.moves import input
from picklable_itertools.extras import equizip
from theano import tensor, Op

import blocks.roles
from blocks.bricks import Tanh, Initializable, Linear, Brick
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import (
    recurrent, BaseRecurrent, SimpleRecurrent, Bidirectional)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks_extras.bricks.sequence_generator2 import (
    SequenceGenerator, SoftmaxReadout, Feedback)
from blocks_extras.bricks.attention2 import AttentionRecurrent
from blocks.config import config
from blocks.graph import ComputationGraph
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.datasets import OneBillionWord, TextFile
from fuel.schemes import ConstantScheme
from blocks.serialization import load_parameters
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.utils import dict_union

from blocks.search import BeamSearch

config.recursion_limit = 100000
floatX = theano.config.floatX
logger = logging.getLogger(__name__)

# Dictionaries
all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '<UNK>'] +
             [' ', '<S>', '</S>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}

COPY = 0
INSERTION = 1
DELETION = 2
SUBSTITUTION = 3

INFINITY = 10 ** 9


def reverse_words(sample):
    sentence = sample[0]
    result = []
    word_start = -1
    for i, code in enumerate(sentence):
        if code >= char2code[' ']:
            if word_start >= 0:
                result.extend(sentence[i - 1:word_start - 1:-1])
                word_start = -1
            result.append(code)
        else:
            if word_start == -1:
                word_start = i
    return (result,)


def _lower(s):
    return s.lower()


def _transpose(data):
    return tuple(array.T for array in data)


def _filter_long(data):
    return len(data[0]) <= 100

def _is_nan(log):
    return math.isnan(log.current_row['total_gradient_norm'])

def _edit_distance_matrix(y, y_hat):
    """Returns the matrix of edit distances.

    Returns
    -------
    dist : numpy.ndarray
        dist[i, j] is the edit distance between the first
    action : numpy.ndarray
        action[i, j] is the action applied to y_hat[j - 1]  in a chain of
        optimal actions transducing y_hat[:j] into y[:i].
        i characters of y and the first j characters of y_hat.

    """
    dist = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
    action = dist.copy()
    for i in xrange(len(y) + 1):
        dist[i][0] = i
    for j in xrange(len(y_hat) + 1):
        dist[0][j] = j

    for i in xrange(1, len(y) + 1):
        for j in xrange(1, len(y_hat) + 1):
            if y[i - 1] != y_hat[j - 1]:
                cost = 1
            else:
                cost = 0
            insertion_dist = dist[i - 1][j] + 1
            deletion_dist = dist[i][j - 1] + 1
            substitution_dist = dist[i - 1][j - 1] + 1 if cost else INFINITY
            copy_dist = dist[i - 1][j - 1] if not cost else INFINITY
            best = min(insertion_dist, deletion_dist,
                       substitution_dist, copy_dist)

            dist[i][j] = best
            if best == insertion_dist:
                action[i][j] = action[i - 1][j]
            if best == deletion_dist:
                action[i][j] = DELETION
            if best == substitution_dist:
                action[i][j] = SUBSTITUTION
            if best == copy_dist:
                action[i][j] = COPY

    return dist, action


def edit_distance(y, y_hat):
    """Edit distance between two sequences.

    Parameters
    ----------
    y : str
        The groundtruth.
    y_hat : str
        The recognition candidate.

   the minimum number of symbol edits (i.e. insertions,
    deletions or substitutions) required to change one
    word into the other.

    """
    return _edit_distance_matrix(y, y_hat)[0][-1, -1]


def trim(y, mask):
    try:
        return y[:mask.index(0.)]
    except ValueError:
        return y


class EditDistanceOp(Op):
    __props__ = ()


    def perform(self, node, inputs, output_storage):
        prediction, prediction_mask, groundtruth, groundtruth_mask = inputs
        if (groundtruth.ndim != 2 or prediction.ndim != 2
                or groundtruth.shape[1] != prediction.shape[1]):
            raise ValueError
        batch_size = groundtruth.shape[1]

        results = numpy.zeros_like(prediction[:, :, None])
        for index in range(batch_size):
            y = trim(list(groundtruth[:, index]),
                     list(groundtruth_mask[:, index]))
            y_hat = trim(list(prediction[:, index]),
                         list(prediction_mask[:, index]))
            results[len(y_hat) - 1, index, 0] = edit_distance(y, y_hat)

        output_storage[0][0] = results

    def grad(self, *args, **kwargs):
        return theano.gradient.disconnected_type()

    def make_node(self, prediction, prediction_mask,
                  groundtruth, groundtruth_mask):
        prediction = tensor.as_tensor_variable(prediction)
        prediction_mask = tensor.as_tensor_variable(prediction_mask)
        groundtruth = tensor.as_tensor_variable(groundtruth)
        groundtruth_mask = tensor.as_tensor_variable(groundtruth_mask)
        return theano.Apply(
            self, [prediction, prediction_mask,
                   groundtruth, groundtruth_mask], [tensor.ltensor3()])


class EditDistanceReward(Brick):

    def __init__(self, **kwargs):
        super(EditDistanceReward, self).__init__(**kwargs)
        self.op = EditDistanceOp()

    @application
    def apply(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask):
        return -self.op(prediction, prediction_mask,
                        groundtruth, groundtruth_mask)


class ReinforceReadout(SoftmaxReadout):

    def __init__(self, reward_brick, baselines, **kwargs):
        super(ReinforceReadout, self).__init__(**kwargs)
        self.reward_brick = reward_brick
        self.baselines = baselines
        self.children += [self.reward_brick]

        # The inputs of "costs" should be automagically set in
        # SoftmaxReadout.__init__. But we should add "baselines"
        self.costs.inputs += [self.baselines]

    @application
    def costs(self, application_call, prediction, prediction_mask,
              groundtruth, groundtruth_mask,
              **all_states):
        rewards = self.reward_brick.apply(prediction, prediction_mask,
                                          groundtruth, groundtruth_mask)

        # Baseline error part
        baselines = all_states.pop(self.baselines)
        future_rewards = rewards.cumsum(axis=0)[::-1]
        centered_future_rewards = (future_rewards - baselines).sum(axis=-1)
        costs = (centered_future_rewards ** 2).sum(axis=0)
        application_call.add_auxiliary_variable(costs, name='baseline_errors')
        # The gradient of this will be the REINFORCE 1-sample
        # gradient estimate
        log_probs = self.all_scores(prediction, **all_states)
        if not prediction_mask:
            prediction_mask = 1
        costs += (centered_future_rewards * (-log_probs)
                 * prediction_mask).sum(axis=0)
        return costs


class WithBaseline(BaseRecurrent, Initializable):
    def __init__(self, recurrent, state_to_use,
                 baselines='baselines',
                 initial_baselines='initial_baselines',
                 **kwargs):
        super(WithBaseline, self).__init__(**kwargs)

        self.recurrent = recurrent
        self.state_to_use = state_to_use
        self.baselines = baselines
        self.initial_baselines = initial_baselines

        self.apply.sequences = recurrent.apply.sequences
        self.apply.states = recurrent.apply.states + [self.baselines]
        self.apply.contexts = (
            recurrent.apply.contexts + [self.initial_baselines])
        self.apply.outputs = recurrent.apply.outputs + [self.baselines]
        self.initial_states.outputs = self.apply.outputs

        self.baseline_predictor = Linear(output_dim=1, name='baseline_predictor')
        self.children = [self.recurrent, self.baseline_predictor]

    def _push_allocation_config(self):
        self.baseline_predictor.input_dim = self.recurrent.get_dim(
            self.state_to_use)

    @recurrent
    def apply(self, **kwargs):
        # We have to have baselines as a state to work nicely with
        # ReinforceReadout.
        kwargs.pop('baselines')
        initial_baselines = kwargs.pop(self.initial_baselines)
        states = theano.gradient.disconnected_grad(kwargs[self.state_to_use])
        baselines = initial_baselines + self.baseline_predictor.apply(states)
        outputs = (self.recurrent.apply(iterate=False, as_list=True, **kwargs)
                   + [baselines])
        return outputs

    def get_dim(self, name):
        if name == self.baselines:
            return 1
        if name == self.initial_baselines:
            return 1
        return self.recurrent.get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        return (self.recurrent.initial_states(batch_size, *args, **kwargs)
                + [kwargs[self.initial_baselines]])


class WordReverser(Initializable):
    """The top brick.

    It is often convenient to gather all bricks of the model under the
    roof of a single top brick.

    """
    def __init__(self, dimension, alphabet_size, **kwargs):
        super(WordReverser, self).__init__(**kwargs)
        encoder = Bidirectional(
            SimpleRecurrent(dim=dimension, activation=Tanh()))
        fork = Fork([name for name in encoder.prototype.apply.sequences
                    if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = [encoder.prototype.get_dim(name) for name in fork.input_names]
        lookup = LookupTable(alphabet_size, dimension)

        recurrent = SimpleRecurrent(
            activation=Tanh(),
            dim=dimension, name="recurrent")
        attention = SequenceContentAttention(
            state_names=recurrent.apply.states,
            attended_dim=2 * dimension, match_dim=dimension, name="attention")
        recurrent_att = AttentionRecurrent(recurrent, attention)
        recurrent_att_baselined = WithBaseline(
            recurrent_att, state_to_use=recurrent.apply.states[0])
        readout = ReinforceReadout(
            EditDistanceReward(),
            dim=alphabet_size,
            merged_states=[recurrent.apply.states[0],
                           attention.take_glimpses.outputs[0]],
            baselines='baselines',
            name="readout")
        feedback = Feedback(recurrent.apply.sequences[:-1],
                            embedding=LookupTable(alphabet_size, dimension))
        generator = SequenceGenerator(
            recurrent_att_baselined, readout, feedback,
            name="generator")
        baseline_readout = Linear(2 * dimension, 1)

        self.fork = fork
        self.lookup = lookup
        self.encoder = encoder
        self.generator = generator
        self.baseline_readout = baseline_readout
        self.children = [fork, lookup, encoder, generator, baseline_readout]

    def mask_for_prediction(self, prediction):
        prediction_mask = tensor.lt(
            tensor.cumsum(tensor.eq(prediction, char2code['</S>'])
                          .astype(theano.config.floatX), axis=0),
            1).astype(floatX)
        prediction_mask = tensor.roll(prediction_mask, 1, 0)
        prediction_mask = tensor.set_subtensor(
            prediction_mask[0, :], tensor.ones_like(prediction_mask[0, :]))
        return prediction_mask

    @application
    def costs(self, chars, chars_mask, targets, targets_mask):
        attended = self.encoder.apply(
            **dict_union(
                self.fork.apply(self.lookup.apply(chars), as_dict=True),
                mask=chars_mask))
        initial_baselines = (
            self.baseline_readout.apply(
                theano.gradient.disconnected_grad(attended)) *
                theano.gradient.disconnected_grad(chars_mask)[:, :, None]).sum(axis=0)
        prediction = theano.gradient.disconnected_grad(
            self.generator.generate(
                n_steps=2 * chars.shape[0], batch_size=chars.shape[1],
                attended=attended,
                attended_mask=chars_mask,
                initial_baselines=initial_baselines)[0])
        prediction_mask = self.mask_for_prediction(prediction)
        return self.generator.costs(
            prediction=prediction, prediction_mask=prediction_mask,
            groundtruth=targets, groundtruth_mask=targets_mask,
            attended=attended,
            attended_mask=chars_mask,
            initial_baselines=initial_baselines)

    @application
    def generate(self, chars):
        return self.generator.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=self.encoder.apply(
                **dict_union(
                    self.fork.apply(self.lookup.apply(chars), as_dict=True))),
            attended_mask=tensor.ones(chars.shape))


class Strings(MonitoredQuantity):

    def initialize(self):
        self.result = None

    def accumulate(self, string, mask):
        self.result = [
            "".join(
                [code2char[code] for code in trim(
                 list(string[:, i]), list(mask[:, i]))])
            for i in range(string.shape[1])]

    def readout(self):
        return self.result


class Baselines(MonitoredQuantity):

    def initialize(self):
        self.result = None

    def accumulate(self, baselines, mask):
        self.result = [
            trim(list(baselines[:, i].squeeze()),
                 list(mask[:, i]))
            for i in range(baselines.shape[1])]

    def readout(self):
        return self.result


def main(mode, save_path, num_batches,
         print_frequency=1, test_values=True, data_path=None):
    reverser = WordReverser(100, len(char2code), name="reverser")

    if mode == "train":
        # Data processing pipeline
        dataset_options = dict(dictionary=char2code, level="character",
                               preprocess=_lower)
        if data_path:
            dataset = TextFile(data_path, **dataset_options)
        else:
            dataset = OneBillionWord("training", [99], **dataset_options)
        data_stream = dataset.get_example_stream()
        data_stream = Filter(data_stream, _filter_long)
        data_stream = Mapping(data_stream, reverse_words,
                              add_sources=("targets",))
        data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
        data_stream = Padding(data_stream)
        data_stream = Mapping(data_stream, _transpose)

        # Initialization settings
        reverser.weights_init = IsotropicGaussian(0.1)
        reverser.biases_init = Constant(0.0)
        reverser.push_initialization_config()
        reverser.encoder.weights_init = Orthogonal()
        # It is very ugly, but so far we do not have recurrent_weights_init
        reverser.generator.recurrent.recurrent.transition.weights_init = Orthogonal()
        reverser.initialize()

        # Build the cost computation graph
        chars = tensor.lmatrix("features")
        chars_mask = tensor.matrix("features_mask")
        targets = tensor.lmatrix("targets")
        targets_mask = tensor.matrix("targets_mask")

        if test_values:
            bos = char2code['<S>']
            eos = char2code['</S>']
            spc = char2code[' ']
            chars.tag.test_value = numpy.array(
                [[bos, 3, 4, 5, eos, 0, 0, 0, 0  ],
                 [bos, 1, 2, 3, spc, 4, 5, 6, eos]]).T
            chars_mask.tag.test_value = numpy.array(
                [[1  , 1, 1, 1, 1,   0, 0, 0, 0],
                 [1  , 1, 1, 1, 1,   1, 1, 1, 1]], dtype=theano.config.floatX).T
            targets.tag.test_value = numpy.array(
                [[bos, 5, 4, 3, eos, 0, 0, 0, 0  ],
                 [bos, 3, 2, 1, spc, 6, 5, 4, eos]]).T
            targets_mask.tag.test_value = chars_mask.tag.test_value
            theano.config.compute_test_value = 'warn'
            theano.config.print_test_value = True
        # Smart averaging of the training cost makes us immune
        # to batches of different size
        batch_cost = reverser.costs(
            chars, chars_mask, targets, targets_mask).sum()
        batch_size = chars.shape[1].copy(name="batch_size")
        cost = aggregation.mean(batch_cost, batch_size)
        cost.name = "sequence_log_likelihood"
        logger.info("Cost graph is built")

        # Give an idea of what's going on
        model = Model(cost)
        parameters = model.get_parameter_dict()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in parameters.items()],
                        width=120))

        # Define the training algorithm.
        cg = ComputationGraph(cost)
        assert cg.updates
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]))
        algorithm.updates += cg.updates.items()

        # Fetch variables useful for debugging
        generator = reverser.generator
        energies, = VariableFilter(
            applications=[generator.readout._merge],
            name_regex="output")(cg.variables)
        activations, = VariableFilter(
            applications=[generator.recurrent.apply],
            name=generator.recurrent.apply.states[0])(cg)
        rewards, = VariableFilter(
            bricks=[EditDistanceReward],
            roles=[blocks.roles.OUTPUT])(cg.variables)
        baselines, = VariableFilter(
            applications=[generator.costs], name='baselines')(cg)
        baseline_error, = VariableFilter(
            bricks=[ReinforceReadout], name='baseline_errors')(cg)
        mean_baseline_error = baseline_error.mean().copy('mean_baseline_error')
        prediction, = VariableFilter(
            applications=[generator.costs], name='prediction')(cg)
        prediction_mask, = VariableFilter(
            applications=[generator.costs], name='prediction_mask')(cg)
        mean_reward = rewards.sum(axis=0).mean().copy('mean_reward')
        max_length = chars.shape[0].copy(name="max_length")
        cost_per_character = aggregation.mean(
            batch_cost, batch_size * max_length).copy(
                name="character_log_likelihood")
        min_energy = energies.min().copy(name="min_energy")
        max_energy = energies.max().copy(name="max_energy")
        mean_activation = abs(activations).mean().copy(
                name="mean_activation")
        observables = [
            cost,
            mean_reward, mean_baseline_error,
            #min_energy, max_energy, mean_activation,
            batch_size, max_length, cost_per_character,
            algorithm.total_step_norm, algorithm.total_gradient_norm]
        observables += [Strings(requires=[prediction, prediction_mask],
                                   name='prediction'),
                        Strings(requires=[targets, targets_mask],
                                   name='groundtruth'),
                        Baselines(requires=[baselines, prediction_mask],
                                  name='baselines')]
        # for name, parameter in parameters.items():
        #     observables.append(parameter.norm(2).copy(name + "_norm"))
        #     observables.append(algorithm.gradients[parameter].norm(2).copy(
        #         name + "_grad_norm"))

        # Construct the main loop and start training!
        main_loop = MainLoop(
            model=model,
            data_stream=data_stream,
            algorithm=algorithm,
            extensions=[
                Timing(every_n_batches=print_frequency),
                TrainingDataMonitoring(
                    observables, prefix="average", every_n_batches=print_frequency),
                TrainingDataMonitoring(
                    [algorithm.total_gradient_norm], after_batch=True),
                FinishAfter(after_n_batches=num_batches)
                # This shows a way to handle NaN emerging during
                # training: simply finish it.
                .add_condition(["after_batch"], _is_nan),
                # Saving the model and the log separately is convenient,
                # because loading the whole pickle takes quite some time.
                Checkpoint(save_path, every_n_batches=2000,
                           save_separately=["model", "log"]),
                Printing(every_n_batches=print_frequency)])
        main_loop.run()
    elif mode == "sample" or mode == "beam_search":
        chars = tensor.lmatrix("input")
        generated = reverser.generate(chars)
        model = Model(generated)
        logger.info("Loading the model..")
        model.set_parameter_values(load_parameters(open(save_path)))

        def generate(input_):
            """Generate output sequences for an input sequence.

            Incapsulates most of the difference between sampling and beam
            search.

            Returns
            -------
            outputs : list of lists
                Trimmed output sequences.
            costs : list
                The negative log-likelihood of generating the respective
                sequences.

            """
            if mode == "beam_search":
                samples, = VariableFilter(
                    applications=[reverser.generator.generate], name="outputs")(
                        ComputationGraph(generated[1]))
                # NOTE: this will recompile beam search functions
                # every time user presses Enter. Do not create
                # a new `BeamSearch` object every time if
                # speed is important for you.
                beam_search = BeamSearch(samples)
                outputs, costs = beam_search.search(
                    {chars: input_}, char2code['</S>'],
                    3 * input_.shape[0])
            else:
                samples, scores =  model.get_theano_function()(input_)[:2]
                outputs = list(samples.T)
                costs = list(-scores.T)
                masks = samples.copy()
                for i in range(len(outputs)):
                    outputs[i] = list(outputs[i])
                    try:
                        true_length = outputs[i].index(char2code['</S>']) + 1
                    except ValueError:
                        true_length = len(outputs[i])
                    outputs[i] = outputs[i][:true_length]
                    costs[i] = costs[i][:true_length].sum()
                    masks[:, i] = 0.
                    masks[:true_length, i] = 1.
                # Sanity check
                args = [input_, numpy.ones_like(input_).astype('float32'),
                        samples, masks.astype('float32')]
                args = [tensor.as_tensor_variable(arg) for arg in args]
                costs2 = reverser.costs(*args).eval()
                print(costs, costs2)
            return outputs, costs

        while True:
            try:
                line = input("Enter a sentence\n")
                message = ("Enter the number of samples\n" if mode == "sample"
                        else "Enter the beam size\n")
                batch_size = int(input(message))
            except EOFError:
                break
            except Exception:
                traceback.print_exc()
                continue

            encoded_input = [char2code.get(char, char2code["<UNK>"])
                             for char in line.lower().strip()]
            encoded_input = ([char2code['<S>']] + encoded_input +
                             [char2code['</S>']])
            print("Encoder input:", encoded_input)
            target = reverse_words((encoded_input,))[0]
            print("Target: ", target)

            samples, costs = generate(
                numpy.repeat(numpy.array(encoded_input)[:, None],
                             batch_size, axis=1))
            messages = []
            for sample, cost in equizip(samples, costs):
                message = "({})".format(cost)
                message += "".join(code2char[code] for code in sample)
                if sample == target:
                    message += " CORRECT!"
                messages.append((cost, message))
            messages.sort(key=operator.itemgetter(0), reverse=True)
            for _, message in messages:
                print(message)
