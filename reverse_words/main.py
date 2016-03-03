from __future__ import print_function
import logging
import pprint
import math
import numpy
import traceback
import operator

import theano
from theano import tensor
from theano.gradient import disconnected_grad
from six.moves import input
from picklable_itertools.extras import equizip

import blocks.roles
from blocks.select import Selector
from blocks.theano_expressions import l2_norm
from blocks.bricks import Tanh, Initializable, Linear, Brick
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import (
    recurrent, BaseRecurrent, SimpleRecurrent, Bidirectional, GatedRecurrent)
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
from blocks.algorithms import (GradientDescent, Scale, Restrict,
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

from reverse_words.edit_distance import trim, EditDistanceOp

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
bos = char2code['<S>']
eos = char2code['</S>']
spc = char2code[' ']

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

def copy_words(sample):
    return (sample[0],)

def _shorten(sample):
    text, = sample
    if len(text) > 5:
        text = text[:5]
        text[-1] = char2code['</S>']
    return (text,)

def _lower(s):
    return s.lower()


def _transpose(data):
    return tuple(array.T for array in data)


def _filter_long(data):
    return len(data[0]) <= 100

def _is_nan(log):
    return math.isnan(log.current_row['total_gradient_norm'])

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

    def __init__(self, reward_brick, baselines, entropy=None, **kwargs):
        super(ReinforceReadout, self).__init__(**kwargs)
        self.reward_brick = reward_brick
        self.baselines = baselines
        self.entropy_coof = entropy
        self.children += [self.reward_brick]

        # The inputs of "costs" should be automagically set in
        # SoftmaxReadout.__init__. But we should add "baselines"
        self.costs.inputs += [self.baselines]

    @application
    def costs(self, application_call, prediction, prediction_mask,
              groundtruth, groundtruth_mask,
              **all_states):
        log_probs = self.all_scores(prediction, **all_states)
        rewards = self.reward_brick.apply(prediction, prediction_mask,
                                          groundtruth, groundtruth_mask).sum(axis=-1)
        # Encourage entropy
        sumlogs = disconnected_grad(-log_probs * prediction_mask)
        application_call.add_auxiliary_variable(sumlogs, name='sumlogs')
        if self.entropy_coof:
            rewards += self.entropy_coof * sumlogs

        # Baseline error part
        baselines = all_states.pop(self.baselines).sum(axis=-1)
        future_rewards = rewards[::-1].cumsum(axis=0)[::-1]
        centered_future_rewards = future_rewards - baselines
        baseline_errors = ((centered_future_rewards *
                  disconnected_grad(prediction_mask)) ** 2).sum(axis=0)
        application_call.add_auxiliary_variable(
            baseline_errors, name='baseline_errors')
        # The gradient of this will be the REINFORCE 1-sample
        # gradient estimate
        costs = (disconnected_grad(centered_future_rewards)
                 * (-log_probs)
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
        states = disconnected_grad(kwargs[self.state_to_use])
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
    def __init__(self, dimension, alphabet_size, entropy, **kwargs):
        super(WordReverser, self).__init__(**kwargs)
        encoder = Bidirectional(
            GatedRecurrent(dim=dimension, activation=Tanh()))
        fork = Fork([name for name in encoder.prototype.apply.sequences
                    if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = [encoder.prototype.get_dim(name) for name in fork.input_names]
        lookup = LookupTable(alphabet_size, dimension)

        recurrent = GatedRecurrent(
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
            entropy=entropy,
            dim=alphabet_size,
            merged_states=[recurrent.apply.states[0],
                           attention.take_glimpses.outputs[0]],
            baselines='baselines',
            name="readout")
        feedback = Feedback([name for name in recurrent.apply.sequences
                             if name != 'mask'],
                            embedding=LookupTable(alphabet_size, dimension))
        generator = SequenceGenerator(
            recurrent_att_baselined, readout, feedback,
            name="generator")
        baseline_readout = Linear(2 * dimension, 1, name='baseline_readout')

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
                disconnected_grad(attended)) *
                disconnected_grad(chars_mask)[:, :, None]).sum(axis=0)
        prediction = theano.gradient.disconnected_grad(
            self.generator.generate(
                n_steps=5,#2 * chars.shape[0],
                batch_size=chars.shape[1],
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
        attended = self.encoder.apply(
            **dict_union(
                self.fork.apply(self.lookup.apply(chars), as_dict=True)))
        return self.generator.generate(
            n_steps=3 * chars.shape[0],
            batch_size=chars.shape[1],
            initial_baselines=self.baseline_readout.apply(attended).sum(axis=0),
            attended=attended,
            attended_mask=tensor.ones(chars.shape))


class Strings(MonitoredQuantity):

    def initialize(self):
        self.result = None

    def aggregate(self, string, mask):
        self.result = [
            "".join(
                [code2char[code] for code in trim(
                 list(string[:, i]), list(mask[:, i]))])
            for i in range(string.shape[1])]

    def get_aggregated_value(self):
        return self.result


class Baselines(MonitoredQuantity):

    def initialize(self):
        self.result = None

    def aggregate(self, baselines, mask):
        self.result = [
            trim(list(baselines[:, i].squeeze()),
                 list(mask[:, i]))
            for i in range(baselines.shape[1])]

    def get_aggregated_value(self):
        return self.result


def main(mode, save_path, num_batches,
         print_frequency=1, verbose=False,
         entropy=None,
         test_values=False, data_path=None):
    if test_values:
        theano.config.compute_test_value = 'warn'
        theano.config.print_test_value = True

    reverser = WordReverser(100, len(char2code), entropy, name="reverser")
    generator = reverser.generator

    if mode == "train":
        # Data processing pipeline
        dataset_options = dict(dictionary=char2code, level="character",
                               preprocess=_lower)
        if data_path:
            dataset = TextFile(data_path, **dataset_options)
        else:
            dataset = OneBillionWord("training", [99], **dataset_options)
        data_stream = dataset.get_example_stream()
        #data_stream = Filter(data_stream, _filter_long)
        #data_stream = Mapping(data_stream, reverse_words,
                              #add_sources=("targets",))
        data_stream = Mapping(data_stream, _shorten)
        data_stream = Mapping(data_stream, copy_words,
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

        # Smart averaging of the training cost makes us immune
        # to batches of different size
        batch_cost = reverser.costs(
            chars, chars_mask, targets, targets_mask).sum()
        batch_size = chars.shape[1].copy(name="batch_size")
        cost = aggregation.mean(batch_cost, batch_size)
        cost.name = "cost"
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
        baseline_errors, = VariableFilter(
            bricks=[ReinforceReadout], name='baseline_errors')(cg)
        mean_baseline_error = baseline_errors.mean().copy('mean_baseline_error')
        assert cg.updates
        baseline_parameters = list(Selector([
            reverser.baseline_readout, generator.recurrent.baseline_predictor])
            .get_parameters().values())
        normal_parameters = [p for p in cg.parameters if p not in baseline_parameters]
        algorithm = GradientDescent(
            cost=cost + mean_baseline_error,
            parameters=cg.parameters,
            step_rule=CompositeRule([
                Restrict(CompositeRule([StepClipping(100.0), Scale(0.01)]),
                         normal_parameters),
                Restrict(CompositeRule([Scale(1e-6)]),
                         baseline_parameters)]))
        baseline_gradient_norm = l2_norm(
            [algorithm.gradients[p] for p in baseline_parameters]).copy(
                name='baseline_gradient_norm')
        normal_gradient_norm = l2_norm(
            [algorithm.gradients[p] for p in normal_parameters]).copy(
                name='normal_gradient_norm')

        algorithm.updates += cg.updates.items()

        # Fetch variables useful for debugging
        energies, = VariableFilter(
            applications=[generator.readout._merge],
            name_regex="output")(cg.variables)
        activations, = VariableFilter(
            applications=[generator.recurrent.apply],
            name=generator.recurrent.apply.states[0])(cg)
        rewards, = VariableFilter(
            bricks=[EditDistanceReward],
            roles=[blocks.roles.OUTPUT])(cg.variables)
        rewards = rewards.sum(axis=0).sum(axis=-1).copy(name='rewards')
        initial_baselines, = VariableFilter(
            applications=[generator.costs], name='initial_baselines')(cg)
        initial_baselines = initial_baselines.sum(axis=-1).copy(name='initial_baselines')
        baselines, = VariableFilter(
            applications=[generator.costs], name='baselines')(cg)
        sumlogs, = VariableFilter(
            bricks=[ReinforceReadout], name='sumlogs')(cg)
        prediction, = VariableFilter(
            applications=[generator.costs], name='prediction')(cg)
        prediction_mask, = VariableFilter(
            applications=[generator.costs], name='prediction_mask')(cg)
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
            rewards.mean().copy('mean_reward'),
            sumlogs.mean().copy('entropy'),
            mean_baseline_error,
            reverser.baseline_readout.b.copy(name='baseline_bias'),
            algorithm.gradients[reverser.baseline_readout.b].copy(
                name='baseline_bias_grad'),
            #min_energy, max_energy, mean_activation,
            batch_size, max_length, cost_per_character,
            algorithm.total_step_norm, algorithm.total_gradient_norm,
            baseline_gradient_norm, normal_gradient_norm]
        #observables += [Strings(requires=[prediction, prediction_mask],
                                   #name='prediction'),
                        #Strings(requires=[targets, targets_mask],
                                   #name='groundtruth'),
                        #Baselines(requires=[baselines, prediction_mask],
                                  #name='baselines')]
        # for name, parameter in parameters.items():
        #     observables.append(parameter.norm(2).copy(name + "_norm"))
        #     observables.append(algorithm.gradients[parameter].norm(2).copy(
        #         name + "_grad_norm"))

        # Construct the main loop and start training!
        extensions=[Timing(every_n_batches=print_frequency)]
        if verbose:
            extensions += [
                TrainingDataMonitoring(
                    [Strings(requires=[prediction, prediction_mask],
                            name='prediction'),
                     Strings(requires=[targets, targets_mask],
                            name='groundtruth'),
                     Baselines(requires=[baselines, prediction_mask],
                            name='baselines'),
                     rewards, initial_baselines, mean_baseline_error],
                    after_batch=True)]
        extensions += [
            TrainingDataMonitoring(
                observables, every_n_batches=print_frequency),
            TrainingDataMonitoring(
                [algorithm.total_gradient_norm], after_batch=True),
            FinishAfter(after_n_batches=num_batches)
            # This shows a way to handle NaN emerging during
            # training: simply finish it.
            .add_condition(["after_batch"], _is_nan),
            # Saving the model and the log separately is convenient,
            # because loading the whole pickle takes quite some time.
            Checkpoint(save_path, every_n_batches=200,
                        save_separately=["model", "log"]),
            Printing(every_n_batches=print_frequency)]
        main_loop = MainLoop(
            model=model,
            data_stream=data_stream,
            algorithm=algorithm,
            extensions=extensions)
        main_loop.run()
    elif mode == "sample" or mode == "beam_search":
        chars = tensor.lmatrix("input")
        chars.tag.test_value = numpy.array(
            [[bos, 3, 4, 5, eos, 0, 0, 0, 0  ],
             [bos, 1, 2, 3, spc, 4, 5, 6, eos]]).T

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
                # args = [input_, numpy.ones_like(input_).astype('float32'),
                #        samples, masks.astype('float32')]
                # args = [tensor.as_tensor_variable(arg) for arg in args]
                # costs2 = reverser.costs(*args).eval()
                # print(costs, costs2)
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
