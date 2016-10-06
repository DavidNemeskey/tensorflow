#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Generic language modeling with RNN."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
import glob
import math
import os
import re
import sys
import time

import numpy as np
import tensorflow as tf

from auxiliary import AttrDict
from file_reader import file_reader

BATCH, TIME, VOCAB = range(3)


class LSTMModel(object):
    """An LSTM language model."""
    def __init__(self, params, is_training):
        self.params = params
        self.is_training = is_training

        """This creates the graph."""
        self.data, self.target, self.mask, self.length = self._data()
        self.initial_state, self.prediction, self.state = self._forward()
        self.loss, self.error, self.logprob = \
            self._cost(), self._error(), self._logprob()

        # Optimization is not needed for the validation and testing cases
        if not is_training:
            return
        self.optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
        self.optimize = self._optimize()

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
        num_unrolled where elements are zero or one depending on wheter the
        respective frame is used or a padding frame. The length property sums
        up the mask along the time axis in order to obtain the length of each
        sequence.
        """
        num_unrolled = self.params.num_unrolled
        self.sequence = tf.placeholder(
            self.params.data_type, [None, num_unrolled, self.params.vocabulary])
        data = tf.slice(self.sequence, (0, 0, 0), (-1, num_unrolled - 1, -1))
        target = tf.slice(self.sequence, (0, 1, 0), (-1, -1, -1))
        mask = tf.reduce_max(tf.abs(target), reduction_indices=VOCAB)
        length = tf.reduce_sum(mask, reduction_indices=TIME)  # so /batch
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
        if self.params.embedding == 'yes':
            with tf.device("/cpu:0"):
                data = tf.argmax(self.data, VOCAB)
                embedding = tf.get_variable(
                    'embedding', [self.params.vocabulary, self.params.rnn_hidden],
                    dtype=self.params.data_type)
                self.inputs = tf.nn.embedding_lookup(embedding, data)
        else:
            self.inputs = self.data

        cell = tf.nn.rnn_cell.BasicLSTMCell(self.params.rnn_hidden,
                                            state_is_tuple=True)
        if self.is_training and self.params.keep_prob < 1:
            cell = tf.nn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob=self.params.keep_prob)
        # TODO: max norm. reg.
        if self.params.rnn_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.params.rnn_layers,
                                               state_is_tuple=True)
        initial_state = cell.zero_state(self.params.batch_size,
                                        self.params.data_type)

        print(self.inputs, self.data)
        hidden, state = tf.nn.dynamic_rnn(
            inputs=self.inputs, cell=cell, dtype=self.params.data_type,
            initial_state=initial_state, sequence_length=self.length)
        vocabulary_size = int(self.target.get_shape()[2])
        prediction = self._shared_softmax(hidden, vocabulary_size)
        return initial_state, prediction, state

    def _shared_softmax(self, data, out_size):
        """Computes the shared softmax over all time-steps."""
        num_unrolled = int(data.get_shape()[1])  # time-steps
        in_size = int(data.get_shape()[2])     # vocabulary size
        # TODO or regular sqrt?
        weight = tf.get_variable(
            'softmax_w', [in_size, out_size], dtype=self.params.data_type,
            initializer=tf.truncated_normal_initializer(
                stddev=1 / math.sqrt(in_size), dtype=self.params.data_type))
        bias = tf.get_variable(
            'softmax_b', shape=[out_size], dtype=self.params.data_type,
            initializer=tf.constant_initializer(0.1, dtype=self.params.data_type))
        # Flatten to apply same weights to all time steps.
        flat = tf.reshape(data, [-1, in_size])
        # Softmax just computes the values, doesn't choose the best, so we can
        # just flatten and reshape the matrix
        output = tf.nn.softmax(tf.matmul(flat, weight) + bias)
        output = tf.reshape(output, [-1, num_unrolled, out_size])
        return output

    def _optimize(self):
        gradient = self.optimizer.compute_gradients(self._cost())
        if self.params.gradient_clipping:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient]
        optimize = self.optimizer.apply_gradients(gradient)
        return optimize

    def _cost(self):
        """Cross-entropy loss."""
        prediction = tf.clip_by_value(self.prediction, 1e-10, 1.0)
        cost = self.target * tf.log(prediction)
        cost = -tf.reduce_sum(cost, reduction_indices=VOCAB)
        return self._average(cost)

    def _error(self):
        """This is basically inverse accuracy."""
        error = tf.not_equal(
            tf.argmax(self.prediction, VOCAB), tf.argmax(self.target, VOCAB))
        error = tf.cast(error, tf.float32)
        return self._average(error)

    def _logprob(self):
        """
        The logprob property is new. It describes the probability that our
        model assigned to the correct next character in logarithmic space. This
        is basically the negative cross entropy transformed into logarithmic
        space and averaged there. Converting the result back into linear space
        yields the so-called perplexity, a common measure to evaluate the
        performance of language models.

        The perplexity can even become infinity when the model assigns a zero
        probability to the next character once. We prevent this extreme case by
        clamping the prediction probabilities within a very small positive
        number and one.
        """
        logprob = tf.mul(self.prediction, self.target)
        logprob = tf.reduce_max(logprob, reduction_indices=VOCAB)
        logprob = tf.log(tf.clip_by_value(logprob, 1e-10, 1.0)) / tf.log(2.0)
        return self._average(logprob)

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
        length = tf.maximum(self.length, 1)  # To protect against / 0
        # length = tf.reduce_sum(self.length, 0)  # To protect against / 0
        data = tf.reduce_sum(data, reduction_indices=TIME) / length
        return tf.reduce_mean(data)  # The avg. data / batch

    def run_training(self, train_reader, valid_reader, gpu_memory=None):
        """Runs the training with the texts given."""
        with tf.Session(graph=self.graph, config=self._get_config(gpu_memory)) as sess:
            last_epoch = self._init_or_load_session(sess)
            batches = iter(train_reader)
            print(valid_reader)
            print('Epoch {:2d}                 valid PPL {:5.1f}'.format(
                last_epoch, self._iter_perplexity(sess, valid_reader)))

            valid_ppls = []
            for epoch in range(last_epoch, self.params.epochs + 1):
                logprobs = []
                for _ in range(self.params.epoch_size):
                    logprobs.append(self._optimization(next(batches), sess))
                self.saver.save(
                    sess, os.path.join(self.save_dir, 'model'), epoch)
                train_ppl = self._perplexity(sess, logprobs=logprobs)
                valid_ppl = self._iter_perplexity(sess, valid_reader)
                print('Epoch {:2d} train PPL {:5.1f} valid PPL {:5.1f}'.format(
                    epoch, train_ppl, valid_ppl))
                valid_ppls.append(valid_ppl)
                # Check for overfitting
                if self._stop_early(valid_ppls):
                    return

    def _stop_early(self, valid_ppls):
        """
        Stops early, i.e.
        - checks if we want early stopping and if the PPL of the validation set
          has been detoriating
        - deletes all checkpoints later than the best performing one.
        - return True if we stopped early; False otherwise
        """
        early_stop = self.params.early_stopping
        if (
            early_stop > 0 and
            np.argmin(valid_ppls) < len(valid_ppls) - early_stop
        ):
            checkpoint = tf.train.get_checkpoint_state(self.save_dir)
            all_checkpoints = checkpoint.all_model_checkpoint_paths
            tf.train.update_checkpoint_state(
                self.save_dir, all_checkpoints[-early_stop - 1],
                all_checkpoints[:-early_stop])
            for checkpoint_to_delete in all_checkpoints[-early_stop:]:
                for file_to_delete in glob.glob(checkpoint_to_delete + '*'):
                    os.remove(file_to_delete)
            print('Stopping training due to overfitting; deleted models ' +
                  'after {}'.format(
                      all_checkpoints[-early_stop - 1].rsplit('-', 1)[-1]))
            return True
        else:
            return False

    def run_evaluation(self, test_reader, gpu_memory=None):
        """Computes the perplexity of the test set."""
        with tf.Session(graph=self.graph, config=self._get_config(gpu_memory)) as sess:
            checkpoint = tf.train.get_checkpoint_state(self.save_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                path = checkpoint.model_checkpoint_path
                self.saver.restore(sess, path)
                print('Restored', path)

                test_ppl = self._iter_perplexity(sess, test_reader)
                print('Test set perplexity: {:5.1f}'.format(test_ppl))
            else:
                print('No checkpoint available; train a model first.')

    def _optimization(self, batch, sess):
        """Runs the optimization."""
        logprob, _ = sess.run((self.logprob, self.optimize),
                              {self.sequence: batch})
        if np.isnan(logprob):
            raise Exception('training diverged')
        return logprob

    def _iter_perplexity(self, sess, file_reader):
        """
        I am pretty sure this isn't correct (because of the last, incomplete
        set of batches -- but I am not even sure the _average is correct at
        this point...
        """
        ppls = [self._perplexity(sess, batches=batches)
                for batches in file_reader]
        return np.mean(ppls)

    def _perplexity(self, sess, batches=None, logprobs=None):
        if logprobs is None:
            logprobs = [
                sess.run(self.logprob,
                         {self.sequence: batches[b:b + self.params.batch_size]})
                for b in range(0, len(batches), self.params.batch_size)
            ]
        return 2 ** -(sum(logprobs) / len(logprobs))


def init_or_load_session(sess, save_dir, saver, init):
    """Initiates or loads a session."""
    checkpoint = tf.train.get_checkpoint_state(save_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        path = checkpoint.model_checkpoint_path
        print('Load checkpoint', path)
        saver.restore(sess, path)
        epoch = int(re.search(r'-(\d+)$', path).group(1)) + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        print('Randomly initialize variables')
        sess.run(init)
        epoch = 1
    return epoch


def run_epoch(session, model, epoch_size, data_iter):
    """Runs an epoch on the network."""
    start_time = time.time()
    state = session.run(model.initial_state)  # Is this required with forget=1?
    fetches = {
        'logprob': model.logprob,
        'final_state': model.state
    }
    # For training models, optimization is the main goal; PPL is just fluff
    if model.is_training:
        fetches['optimize'] = model.optimize

    logprobs = []
    for _ in range(epoch_size if not epoch_size > 0 else sys.maxsize):
        # Break if there is no more data -- useful for consume_all
        batch = next(data_iter, None)
        if batch is None:
            break

        feed_dict = {model.sequence: batch}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c  # CEC for layer i
            feed_dict[h] = state[i].h  # hidden for layer i

        vals = session.run(fetches, feed_dict)
        state = vals['final_state']
        logprobs.append(vals['logprob'])

    # TODO I am pretty sure the perplexity model built in is not correct...
    end_time = time.time()
    perplexity = 2 ** -(sum(logprobs) / len(logprobs))
    return perplexity, end_time - start_time

def stop_early(early_stop, valid_ppls, save_dir):
    """
    Stops early, i.e.
    - checks if we want early stopping and if the PPL of the validation set
      has been detoriating
    - deletes all checkpoints later than the best performing one.
    - return True if we stopped early; False otherwise
    """
    if (
        early_stop > 0 and
        np.argmin(valid_ppls) < len(valid_ppls) - early_stop
    ):
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        all_checkpoints = checkpoint.all_model_checkpoint_paths
        tf.train.update_checkpoint_state(
            save_dir, all_checkpoints[-early_stop - 1],
            all_checkpoints[:-early_stop])
        for checkpoint_to_delete in all_checkpoints[-early_stop:]:
            for file_to_delete in glob.glob(checkpoint_to_delete + '*'):
                os.remove(file_to_delete)
        print('Stopping training due to overfitting; deleted models ' +
              'after {}'.format(
                  all_checkpoints[-early_stop - 1].rsplit('-', 1)[-1]))
        return True
    else:
        return False


def parse_arguments():
    parser = ArgumentParser(
        description='Character-based language modeling with RNN.')
    parser.add_argument('train_file', help='the text file to train on.')
    parser.add_argument('valid_file',
                        help='the text file to use as a validation set.')
    parser.add_argument('test_file',
                        help='the text file to use as a test set.')
    parser.add_argument('vocab_file',
                        help='the vocabulary list common to all sets.')
    parser.add_argument('--model-name', '-m', default='RNN CLM',
                        help='the name of the model [RNN CLM].')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='the training batch size [100].')
    # parser.add_argument('--num-unrollings', '-u', type=int, default=10,
    #                     help='unroll the RNN for how many steps [10].')
    parser.add_argument('--num-nodes', '-n', type=int, default=200,
                        help='use how many RNN cells [200].')
    parser.add_argument('--num-unrolled', '-w', type=int, default=20,
                        help='how many steps to unroll the network for [20].')
    parser.add_argument('--rnn-cell', '-c', choices=['rnn', 'lstm', 'gru'],
                        default='lstm', help='the RNN cell to use [lstm].')
    parser.add_argument('--layers', '-L', type=int, default=1,
                        help='the number of RNN laercell to use [lstm].')
    parser.add_argument('--dropout', '-d', type=float, default=None,
                        help='the keep probability of dropout; if not ' +
                             'specified, no dropout is applied.')
    parser.add_argument('--embedding', '-E', choices={'no', 'yes'}, default='yes',
                        help='whether to compute an embedding as well [yes].')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='the default number of epochs [20].')
    parser.add_argument('--epoch-size', '-s', type=int, default=200,
                        help='the default epoch size [200]. The number of '
                             'batches processed in an epoch.')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.02,
                        help='the default learning rate [0.02].')
    parser.add_argument('--gradient-clipping', '-g', type=float, default=None,
                        help='the limit for gradient clipping [None].')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='early stop after the perplexity has been '
                             'detoriating after this many steps. If 0 (the '
                             'default), do not do early stopping.')
    parser.add_argument('--gpu-memory', type=float, default=None,
                        help='limit on the GPU memory ratio [None].')
    args = parser.parse_args()

    return args


def _get_sconfig(gpu_memory):
    """Gets a session configuration object, which sets the gpu_memory limit."""
    if gpu_memory:
        return tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory))
    else:
        return None


def main():
    args = parse_arguments()

    train_reader = file_reader(args.train_file, args.vocab_file,
                               args.batch_size, args.num_unrolled)
    valid_reader = file_reader(args.valid_file, args.vocab_file,
                               args.batch_size, args.num_unrolled, True)
    test_reader = file_reader(args.test_file, args.vocab_file,
                              args.batch_size, args.num_unrolled, True)

    params = AttrDict(
        rnn_hidden=args.num_nodes,
        rnn_layers=args.layers,
        batch_size=args.batch_size,
        num_unrolled=args.num_unrolled,
        keep_prob=args.dropout,
        vocabulary=len(train_reader.vocab_map),
        learning_rate=args.learning_rate,
        gradient_clipping=args.gradient_clipping,
        embedding=args.embedding,
        data_type=tf.float32
    )
    eval_params = AttrDict(params)
    eval_params.batch_size = 1
    eval_params.num_unrolled = 1

    # Model definition
    with tf.Graph().as_default() as graph:
        init_scale = 1 / math.sqrt(args.num_nodes)
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)

        with tf.name_scope('Train'):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                mtrain = LSTMModel(params, is_training=True)
        with tf.name_scope('Valid'):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = LSTMModel(params, is_training=False)
        with tf.name_scope('Test'):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = LSTMModel(eval_params, is_training=False)
        with tf.name_scope('Global_ops'):
            saver = tf.train.Saver(name='saver', max_to_keep=10)
            init = tf.initialize_all_variables()

    # TODO: look into Supervisor
    # The training itself
    with tf.Session(graph=graph, config=_get_sconfig(args.gpu_memory)) as sess:
        # Load the model, or initialize it anew
        save_dir = os.path.join('saves', args.model_name)
        last_epoch = init_or_load_session(sess, save_dir, saver, init)

        valid_ppls = []
        train_iter = iter(train_reader)
        for epoch in range(last_epoch, args.epochs + 1):
            train_ppl = run_epoch(sess, mtrain, args.epoch_size, train_iter)
            valid_ppl = run_epoch(sess, mvalid, 0, iter(valid_reader))
            print('Epoch {:2d} train PPL {:7.3f} valid PPL {:7.3f}'.format(
                epoch, train_ppl, valid_ppl))
            saver.save(sess, os.path.join(save_dir, 'model'), epoch)

            valid_ppls.append(valid_ppl)
            # Check for overfitting
            if stop_early(args.early_stopping, valid_ppls, save_dir):
                break

        test_ppl = run_epoch(sess, mtest, 0, iter(test_reader))
        print('Test perplexity: {:.3f}'.format(test_ppl))


if __name__ == '__main__':
    main()
