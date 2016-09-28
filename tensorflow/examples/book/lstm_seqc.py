#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Sequence classification with LSTM."""
from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
import gzip.open as gopen
import os

import numpy as np
import tensorflow as tf

from attr_dict import AttrDict


def parse_arguments():
    parser = ArgumentParser(
        description='Sequence classification with LSTM.')
    parser.add_argument('sentiment_prefix',
                        help='the sentiment tsv file prefix: the part before '
                             'train/valid/test.')
    parser.add_argument('embedding_file', help='the embedding file in text format.')
    parser.add_argument('--vocab', '-v', help='the sentiment vocabulary. '
                                              'Optional, but reduces memory usage.')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                        help='the training batch size [64].')
    parser.add_argument('--num-unrollings', '-u', type=int, default=10,
                        help='unroll the RNN for how many steps [10].')
    parser.add_argument('--num-nodes', '-n', type=int, default=64,
                        help='use how many RNN cells [64].')
    parser.add_argument('--rnn-cell', '-c', choices=['rnn', 'lstm', 'gru'],
                        default='lstm', help='the RNN cell to use [lstm].')
    parser.add_argument('--iterations', type=int, default=10001,
                        help='the default number of iterations [10001].')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.02,
                        help='the default learning rate [0.02].')
    parser.add_argument('--gradient-clipping', '-g', action='store_true',
                        help='if gradient clipping should be used.')
    parser.add_argument('--print-every', type=int, default=1000,
                        help='print validation set accuracy every this steps'
                             ' [1000].')
    parser.add_argument('--save-every', type=int, default=2000,
                        help='save the model every this steps [2000].')
    args = parser.parse_args()

    if args.rnn_cell == 'gru':
        args.rnn_cell = tf.nn.rnn_cell.GRUCell
    elif args.rnn_cell == 'lstm':
        args.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
    else:
        args.rnn_cell = tf.nn.rnn_cell.BasicRnnCell
    return args


class Embedding(object):
    def __init__(self, embedding_file, filter_vocab):
        with (gopen if embedding_file.endswith('.gz') else open)(embedding_file) as inf:
            self.words, vectors = [], []
            for line in inf:
                word, vector = line.strip().split(' ', 1)
                if filter_vocab is None or word in filter_vocab:
                    if ' ' in vector:  # header
                        self.words.append(word)
                        vectors.append(vector.split(' '))

            self.vectors = np.array([list(map(float, l)) for l in vectors],
                                    dtype=np.float32)
            self.indices = {word: i for i, word in self.words}

    def __call__(self, sequence):
        # data = np.zeros((len(sequence), self.vectors.shape[1]), dtype=np.float32)
        indices = [self.indices.get(w, 0) for w in sequence]
        return self.vectors[indices]

    def dimensions(self):
        return self.vectors.shape[1]


class SequenceClassificationModel(object):
    def __init__(self, params, name='LSTM SC'):
        self.name = name
        self.save_dir = os.path.join('saves', self.name)
        self.params = params
        self.graph = tf.Graph()

        with self.graph.as_default():
            with tf.name_scope('model'):
                self._create_graph()
            with tf.name_scope('global_ops'):
                self.init = tf.initialize_all_variables(name='init')
                self.saver = tf.train.Saver(name='saver')

    def _create_graph(self):
        """This creates the graph."""
        self.data = tf.placeholder(
            tf.float32, [None, self.params.num_unrollings, self.params.dim])
        self.target = tf.placeholder(tf.float32, [None, 2])
        self.length = self._length()
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

    def run_training(self, train_data, train_labels,
                     valid_data, valid_labels):
        sess = tf.Session(graph=self.graph)
        valid_feed = {self.data: valid_data, self.labels: valid_labels}

        # Load the latest training checkpoint, if it exists
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        else:
            sess.run(self.init)
            initial_step = 0

        # The actual training loop
        for step, batch in enumerate(batches):
            feed = {self.data: batch[0], self.target: batch[1]}
            sess.run(self.optimize, feed=feed)
            if step % self.params.print_every == 0:
                accuracy = sess.run(self.accuracy, feed=valid_feed)
                print('{}: {:3.1f}%'.format(step + 1, 100 * accuracy))
 
            # Model checkpoint
            if step % self.params.save_every == 0 and step > 0:
                self.saver.save(sess, self.save_dir, global_step=step + initial_step)

        # Aaand we are done
        self.saver.save(sess, self.save_dir, global_step=step + initial_step)
        accuracy = sess.run(self.accuracy, feed=valid_feed)
        print('Final accuracy: {:3.1f}%'.format(100 * accuracy))
        sess.close()

    def run_test(self, test_data, test_labels):
        sess = tf.Session(graph=self.graph)

        # Load the latest training checkpoint, if it exists
        ckpt = tf.train.get_checkpoint_state(self.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise RuntimeError('No trained models available')

        accuracy = sess.run(self.accuracy,
                            {self.data: test_data, self.labels: test_labels})
        print('Test accuracy: {:3.1f}%'.format(100 * accuracy))
        sess.close()


def main():
    args = parse_arguments()
    params = AttrDict(
        rnn_cell=args.rnn_cell,
        rnn_hidden=args.num_nodes,
        num_unrollings=args.num_unrollings,
        learning_rate=args.learning_rate,
        gradient_clipping=args.gradient_clipping,
        iterations=args.iterations,
        print_every=args.print_every,
        save_every=args.save_every
    )
    model = SequenceClassificationModel(params, 'hello')


if __name__ == '__main__':
    main()
