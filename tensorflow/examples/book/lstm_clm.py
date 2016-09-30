#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Character-based language modeling with RNN."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
import os

import numpy as np
import tensorflow as tf

from auxiliary import AttrDict, openall

BATCH, TIME, VOCAB = range(3)

class CharacterLM(object):
    def __init__(self, params):
        self.name = params.name
        self.params = params
        
        with self.graph.as_default():
            with tf.name_scope('model'):
                self._create_graph()
            with tf.name_scope('global_ops'):
                self.init = tf.initialize_all_variables()
                self.saver = tf.train.Saver(name='saver')

    def _create_graph(self):
        """This creates the graph."""
        self.data, self.target, self.mask, self.length = self._data()
        self.prediction, self.state = self._forward()

    def _data(self):
        """
        As before, we need the maximum length so that we know how much steps to
        unroll.

        This is where we introduce a temporal difference because at timestep t,
        the model should have character s t as input but s t + 1 as target. As
        easy way to obtain data or target is to slice the provided sequence and
        cut away the last or the first frame, respectively.

        We do this slicing using tf.slice() which takes the sequence to slice,
        a tuple of start indices for each dimension, and a tuple of sizes for
        each dimension. For the sizes -1 means to keep all elemnts from the
        start index in that dimension until the end. Since we want to
        slices frames, we only care about the second dimension.

        We also define two properties on the target sequence as we already
        discussed in earler sections: mask is a tensor of size batch_size x
        max_length where elements are zero or one depending on wheter the
        respective frame is used or a padding frame. The length property sums
        up the mask along the time axis in order to obtain the length of each
        sequence.
        """
        max_length = int(self.sequence.get_shape()[1])
        data = tf.slice(self.sequence, (0, 0, 0), (-1, max_length - 1, -1))
        target = tf.slice(self.sequence, (0, 1, 0), (-1, -1, -1))
        mask = tf.reduce_max(tf.abs(self.target), reduction_indices=VOCAB)
        length = tf.reduce_sum(self.mask, reduction_indices=TIME)  # so /batch
        return data, target, mask, length

    def _forward(self):
        """
        The new part about the neural network code above is that we want to get
        both the prediction and the last recurrent activation. Previously, we
        only returned the prediction but the last activation allows us to
        generate sequences more effectively later. Since we only want to
        construct the graph for the recurrent network once, there is a forward
        property that return the tuple of both tensors and prediction and state
        are just there to provide easy access from the outside.
        """
        cell = self.params.rnn_cell(self.params.rnn_hidden)
        if self.params.rnn_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.params.rnn_layers)
        hidden, state = tf.nn.dynamic_rnn(
            inputs=self.data, cell=cell, dtype=tf.float32,
            initial_state=self.initial, sequence_length=self.length)
        vocabulary_size = int(self.target.get_shape()[2])
        prediction = self._shared_softmax(hidden, vocabulary_size)
        return prediction, state

    def _shared_softmax(self, data, out_size):
        """Computes the shared softmax over all time-steps."""
        max_length = int(data.get_shape()[1])  # time-steps
        in_size = int(data.get_shape()[2])     # vocabulary size
        weight = tf.Variable(tf.truncated_normal(
            [in_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        # Flatten to apply same weights to all time steps.
        flat = tf.reshape(data, [-1, in_size])
        # Softmax just computes the values, doesn't choose the best, so we can
        # just flatten and reshape the matrix
        output = tf.nn.softmax(tf.matmul(flat, weight) + bias)
        output = tf.reshape(output, [-1, max_length, out_size])
        return output

    def _cost(self):
        prediction = tf.clip_by_value(self.prediction, 1e-10, 1.0)
        cost = self.target * tf.log(prediction)
        cost = -tf.reduce_sum(cost, reduction_indices=VOCAB)
        return self._average(cost)

    def _average(self, data):
        """
        All the three properties above are averaged over the frames of all
        sequences. With fixed-length sequences, this would be a single
        tf.reduce_mean(), but as we work with variable-length sequences, we
        have to be a bit more careful. First, we mask out padding frames by
        multiplying with the mask. Then we aggregate along the frame size.
        Because the Predictive coding three functions above all multiply with
        the target, each frame has just one element set and we use
        tf.reduce_sum() to aggregate each frame into a scalar.

        Next, we want to average along the frames of each sequence using the
        actual sequence length. To protect against division by zero in case of
        empty sequences, we use the maximum of each sequence length and one.
        Finally, we can use tf.reduce_mean() to average over the examples in
        the batch.
        """
        data *= self.mask
        length = tf.reduce_max(self.length, 1)  # To protect against / 0
        data = tf.reduce_sum(data, reduction_indices=TIME) / length
        return tf.reduce_mean(data)  # The avg. data / batch
