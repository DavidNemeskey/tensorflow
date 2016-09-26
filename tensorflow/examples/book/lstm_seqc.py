#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Sequence classification with LSTM."""
from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
import gzip.open as gopen

import numpy as np


def parse_arguments():
    parser = ArgumentParser(
        description='Sequence classification with LSTM.')
    parser.add_argument('sentiment_data', help='the sentiment tsv file.')
    parser.add_argument('embedding_file', help='the embedding file in text format.')
    parser.add_argument('--vocab', '-v', help='the sentiment vocabulary. '
                                              'Optional, but reduces memory usage.')
    parser.add_argument('--valid-size', type=int, default=1000,
                        help='random set of words to evaluate similarity on [1000].')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='the training batch size [64].')
    parser.add_argument('--num-unrollings', '-u', type=int, default=10,
                        help='unroll the LSTM for how many steps [10].')
    parser.add_argument('--num-nodes', '-n', type=int, default=64,
                        help='use how many LSTM cells [64].')
    parser.add_argument('--iterations', type=int, default=100001,
                        help='the default number of iterations [130001].')
    args = parser.parse_args()


class Embedding(object):
    def __init__(self, embedding_file, filter_vocab):
        with (gopen if embedding_file.endswith('.gz') else open)(embedding_file) as inf:
            self.words, vectors = [], []
            for line in inf:
                word, vector = line.strip().split(' ', 1)
                if filter_vocab is None or word in filter_vocab:
                    if ' ' in vector:  # header
                        words.append(word)
                        vectors.append(vector.split(' '))

            self.vectors = np.array([list(map(float, l)) for l in vectors],
                                    dtype=np.float32)
            self.indices = {words: i for i, word in self.words}

    def __call__(self, sequence):
        #data = np.zeros((len(sequence), self.vectors.shape[1]), dtype=np.float32)
        indices = [self.indices.get(w, 0) for w in sequence]
        return self.vectors[indices]

    def dimensions(self):
        return self.vectors.shape[1]


class SequenceClassificationModel(object):
    def __init__(self, params):
        self.params = params
        self.length = self._length()
        self.data = tf.placeholder(tf.float32, [None, params.length, params.dim])
        self.target = tf.placeholder(tf.float32, [None, 2])
        self.prediction = self._prediction()
        self.optimize = self._optimize()
        self.accuracy = self._accuracy()

    def _length(self):
        """
        First, we obtain the lengths of sequences in the current data batch. We
        need this since the data comes as a single tensor, padded with zero
        vectors to the longest review length.

        Instead of keeping track of the sequence lengths of every review, we
        just compute it dynamically in TensorFlow. To get the length per
        sequence, we collapse the word vectors using the maximum on the
        absolute values. The resulting scalars will be zero for zero vectors
        and larger than zero for any real word vector. We then discretize these
        values to zero or one using tf.sign() and sum up the results along the
        time steps to obtain the length of each sequence. The resulting tensor
        has the length of batch size and contains a scalar length for each
        sequence.
        """
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        return tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)

    def _prediction(self):
        """The recurrent network."""
        # The RNN
        output, _ = tf.nn.dynamic_rnn(
            cell=self.params.rnn_cell(self.params.rnn_hidden),
            inputs=self.data, sequence_length=self.length, dtype=tf.float32
        )
        last = self._last_relevant(output, self.length)

        # Softmax
        num_classes = int(self.target.get_shape()[1])
        weight = tf.Variable(tf.truncated_normal(
            [self.params.rnn_hidden, num_classes], stddev=0.01))
        # TODO: Glorot: 2 / (nin+nout)
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @staticmethod
    def _last_relevant(output, length):
        """
        As of now, TensorFlow only supports indexing along the first dimension,
        using tf.gather() . We thus flatten the first two dimensions of the
        output activations from their shape of sequences x time_steps x
        word_vectors and construct an index into this resulting tensor. The
        index takes into account the start indices for each sequence in the flat
        tensor and adds the sequence length to it. Actually, we only add
        length - 1 so that we select the last valid time step.
        """
        # WTF is this?
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    def _loss(self):
        """Cross entropy loss."""
        return -tf.reduce_sum(self.target * tf.lot(self.prediction))

    def _optimize(self):
        optimizer = tf.train.AdagradOptimizer(self.params.learning_rate)
        gradient = optimizer.compute_gradients(self._loss())
        if self.params.gradient_clipping:
            limit = self.params.gradient_clipping
            gradient = [(tf.clip_by_value(g, -limit, limit), v)
                        if g is not None else (None, v)
                        for g, v in gradient]
        optimize = optimizer.apply_gradients(gradient)
        return optimize

    def _accuracy(self):
        mistakes = tf.equal(tf.argmax(self.target, 1),
                            tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def main():
    pass


if __name__ == '__main__':
    main()
