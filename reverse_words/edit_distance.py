import numpy

import theano
from theano import tensor, Op


COPY = 0
INSERTION = 1
DELETION = 2
SUBSTITUTION = 3

INFINITY = 10 ** 9


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


