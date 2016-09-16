#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Exercises for assignment 6 -- LSTM."""
from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
from itertools import count, dropwhile
import random
import string

import numpy as np
import tensorflow as tf

from text_data import read_as_string


def parse_arguments():
    parser = ArgumentParser(
        description='Exercises for assignment 5 -- word2vec.')
    parser.add_argument('exercise', choices=['0', '1'],
                        help='the exercise to run.')
    parser.add_argument('data_file', help='the pickled data file.')
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

    return (args.exercise, args.data_file, args.valid_size,
            args.batch_size, args.num_unrollings, args.num_nodes,
            args.iterations)


def split_train_valid(text, valid_size):
    split = next(dropwhile(lambda i: not text[i].isspace(), count(valid_size)))
    valid_text = text[:split]
    train_text = text[split + 1:]
    train_size = len(train_text)
    print(train_size, train_text[:64])
    print(valid_size, valid_text[:64])
    return train_text, valid_text


class VocabConverter(object):
    vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
    first_letter = ord(string.ascii_lowercase[0])

    @staticmethod
    def char2id(char):
        if char in string.ascii_lowercase:
            return ord(char) - VocabConverter.first_letter + 1
        elif char == ' ':
            return 0
        else:
            print('Unexpected character: %s' % char)
        return 0

    @staticmethod
    def id2char(dictid):
        return chr(dictid + VocabConverter.first_letter - 1) if dictid > 0 else ' '


VC = VocabConverter


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, VC.vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, VC.char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """
    Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation.
    """
    return [VC.id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """
    Convert a sequence of batches back into their (most likely) string
    representation.
    """
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """
    Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, VC.vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, VC.vocabulary_size])
    return b/np.sum(b, 1)[:, None]


def problem_0(batch_size, num_unrollings, num_nodes, iterations,
              train_batches, valid_batches, valid_size):
    graph = tf.Graph()
    with graph.as_default():
        # Parameters:
        # Input gate: input, previous output, and bias.
        ix = tf.Variable(tf.truncated_normal([VC.vocabulary_size, num_nodes], -0.1, 0.1))
        im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ib = tf.Variable(tf.zeros([1, num_nodes]))
        # Forget gate: input, previous output, and bias.
        fx = tf.Variable(tf.truncated_normal([VC.vocabulary_size, num_nodes], -0.1, 0.1))
        fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        fb = tf.Variable(tf.zeros([1, num_nodes]))
        # Memory cell: input, state and bias.
        cx = tf.Variable(tf.truncated_normal([VC.vocabulary_size, num_nodes], -0.1, 0.1))
        cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output, and bias.
        ox = tf.Variable(tf.truncated_normal([VC.vocabulary_size, num_nodes], -0.1, 0.1))
        om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        ob = tf.Variable(tf.zeros([1, num_nodes]))
        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, VC.vocabulary_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([VC.vocabulary_size]))

        # Definition of the cell computation.
        def lstm_cell(i, o, state):
            """
            Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between
            the previous state and the gates.
            """
            input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
            update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state

        # Input data.
        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(
                tf.placeholder(tf.float32, shape=[batch_size, VC.vocabulary_size]))
        train_inputs = train_data[:num_unrollings]
        train_labels = train_data[1:]  # labels are inputs shifted by one time step.

        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            # Classifier.
            logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits, tf.concat(0, train_labels)))

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.float32, shape=[1, VC.vocabulary_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(
            sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

    # Training
    summary_frequency = 100

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        mean_loss = 0
        for step in range(iterations):
            batches = train_batches.next()
            feed_dict = dict()
            for i in range(num_unrollings + 1):
                feed_dict[train_data[i]] = batches[i]
            _, l, predictions, lr = session.run(
                [optimizer, loss, train_prediction, learning_rate],
                feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step %d: %f learning rate: %f' %
                      (step, mean_loss, lr))
                mean_loss = 0
                labels = np.concatenate(list(batches)[1:])
                print('Minibatch perplexity: %.2f' % float(
                    np.exp(logprob(predictions, labels))))
                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    for _ in range(5):
                        feed = sample(random_distribution())
                        sentence = characters(feed)[0]
                        reset_sample_state.run()
                        for _ in range(79):
                            prediction = sample_prediction.eval({sample_input: feed})
                            feed = sample(prediction)
                            sentence += characters(feed)[0]
                        print(sentence)
                    print('=' * 80)

                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_size):
                    b = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: b[0]})
                    valid_logprob = valid_logprob + logprob(predictions, b[1])
                print('Validation set perplexity: %.2f' % float(np.exp(
                    valid_logprob / valid_size)))


def problem_1(batch_size, num_unrollings, num_nodes, iterations,
              train_batches, valid_batches, valid_size):
    """
    You might have noticed that the definition of the LSTM cell involves 4
    matrix multiplications with the input, and 4 matrix multiplications with the
    output. Simplify the expression by using a single matrix multiply for each,
    and variables that are 4 times larger."""
    graph = tf.Graph()
    with graph.as_default():
        INPUT, FORGET, OUTPUT = range(3)
        # Parameters:
        # Common parameters for all gates + state update
        Gx = tf.Variable(tf.truncated_normal([VC.vocabulary_size, 4 * num_nodes], -0.1, 0.1))
        Gm = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
        Gb = tf.Variable(tf.zeros([1, 4 * num_nodes]))
        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, VC.vocabulary_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([VC.vocabulary_size]))

        def wslice(tensor, begin=0, size=1):
            return tf.slice(tensor, [0, begin * num_nodes], [-1, size * num_nodes])

        # Definition of the cell computation.
        def lstm_cell(i, o, state):
            """
            Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between
            the previous state and the gates.
            """
            gate_mult = tf.matmul(i, Gx) + tf.matmul(o, Gm) + Gb
            gates = tf.sigmoid(wslice(gate_mult, size=3))
            update = wslice(gate_mult, 2)
            state = wslice(gates, FORGET) * state + wslice(gates, INPUT) * tf.tanh(update)
            return wslice(gates, OUTPUT) * tf.tanh(state), state

        # Input data.
        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(
                tf.placeholder(tf.float32, shape=[batch_size, VC.vocabulary_size]))
        train_inputs = train_data[:num_unrollings]
        train_labels = train_data[1:]  # labels are inputs shifted by one time step.

        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output),
                                      saved_state.assign(state)]):
            # Classifier.
            logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits, tf.concat(0, train_labels)))

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.float32, shape=[1, VC.vocabulary_size])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, num_nodes])),
            saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(
            sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

    # Training
    summary_frequency = 100

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        mean_loss = 0
        for step in range(iterations):
            batches = train_batches.next()
            feed_dict = dict()
            for i in range(num_unrollings + 1):
                feed_dict[train_data[i]] = batches[i]
            _, l, predictions, lr = session.run(
                [optimizer, loss, train_prediction, learning_rate],
                feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step %d: %f learning rate: %f' %
                      (step, mean_loss, lr))
                mean_loss = 0
                labels = np.concatenate(list(batches)[1:])
                print('Minibatch perplexity: %.2f' % float(
                    np.exp(logprob(predictions, labels))))
                if step % (summary_frequency * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)
                    for _ in range(5):
                        feed = sample(random_distribution())
                        sentence = characters(feed)[0]
                        reset_sample_state.run()
                        for _ in range(79):
                            prediction = sample_prediction.eval({sample_input: feed})
                            feed = sample(prediction)
                            sentence += characters(feed)[0]
                        print(sentence)
                    print('=' * 80)

                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_size):
                    b = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: b[0]})
                    valid_logprob = valid_logprob + logprob(predictions, b[1])
                print('Validation set perplexity: %.2f' % float(np.exp(
                    valid_logprob / valid_size)))


def main():
    (exercise, data_file, valid_size, batch_size,
     num_unrollings, num_nodes, iterations) = parse_arguments()

    # Read the text
    text = read_as_string(data_file)
    print('Data size %d' % len(text))

    # Create a small validation set.
    train_text, valid_text = split_train_valid(text, valid_size)
    real_valid_size = len(valid_text)

    # Char -- id conversion
    print(VC.char2id('a'), VC.char2id('z'), VC.char2id(' '), VC.char2id('ï'))
    print(VC.id2char(1), VC.id2char(26), VC.id2char(0))

    train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    print(batches2string(train_batches.next()))
    print(batches2string(train_batches.next()))
    print(batches2string(valid_batches.next()))
    print(batches2string(valid_batches.next()))

    problem_1(batch_size, num_unrollings, num_nodes, iterations,
              train_batches, valid_batches, real_valid_size)


if __name__ == '__main__':
    main()
