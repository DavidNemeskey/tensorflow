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
from lstm_model import LSTMModel


def get_sconfig(gpu_memory):
    """
    Returns a session configuration object that sets the GPU memory limit.
    """
    if gpu_memory:
        return tf.ConfigProto(gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory))
    else:
        return None


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
    parser.add_argument('--max-grad-norm', '-g', type=float, default=None,
                        help='the limit for gradient clipping [None].')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='early stop after the perplexity has been '
                             'detoriating after this many steps. If 0 (the '
                             'default), do not stop early.')
    parser.add_argument('--gpu-memory', type=float, default=None,
                        help='limit on the GPU memory ratio [None].')
    args = parser.parse_args()

    return args


def run_epoch(sess, model, epoch_size, data_iter):
    """Runs an epoch on the network."""
    start_time = time.time()
    state = sess.run(model.initial_state)

    fetches = {
        'logprob': model.logprob,
        'final_state': model.state,
    }
    if model.is_training:
        fetches['optimize'] = model.optimize

    logprobs = []
    for _ in range(epoch_size if epoch_size > 0 else sys.maxsize):
        batch = next(data_iter, None)
        if batch is None:
            break

        feed_dict = {model.sequence: batch}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c  # CEC for layer i
            feed_dict[h] = state[i].h  # hidden for layer i

        vals = sess.run(fetches, feed_dict)
        state = vals['final_state']
        logprobs.append(vals['logprob'])

    end_time = time.time()
    perplexity = 2 ** -(sum(logprobs) / len(logprobs))
    return perplexity, end_time - start_time


def _stop_early(valid_ppls, early_stop, save_dir):
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


def main():
    args = parse_arguments()

    train_reader = file_reader(args.train_file, args.vocab_file,
                               args.batch_size, args.num_unrolled)
    valid_reader = file_reader(args.valid_file, args.vocab_file,
                               args.batch_size, args.num_unrolled, True)
    test_reader = file_reader(args.test_file, args.vocab_file,
                              1, 1, True)

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
        data_type=tf.float32,
    )
    eval_params = AttrDict(params)
    eval_params.batch_size = 1
    eval_params.num_unrolled = 1

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
    with tf.Session(graph=graph, config=get_sconfig(args.gpu_memory)) as sess:
        save_dir = os.path.join('saves', args.model_name)
        last_epoch = init_or_load_session(sess, save_dir, saver, init)
        print('Epoch {:2d}-                 valid PPL {:6.3f}'.format(
            last_epoch, run_epoch(sess, mvalid, 0, iter(valid_reader))[0]))

        valid_ppls = []
        train_iter = iter(train_reader)
        for epoch in range(last_epoch, args.epochs + 1):
            train_ppl, _ = run_epoch(sess, mtrain, args.epoch_size, train_iter)
            valid_ppl, _ = run_epoch(sess, mvalid, 0, iter(valid_reader))
            print('Epoch {:2d} train PPL {:6.3f} valid PPL {:6.3f}'.format(
                epoch, train_ppl, valid_ppl))
            saver.save(sess, os.path.join(save_dir, 'model'), epoch)

            valid_ppls.append(valid_ppl)
            # Check for overfitting
            if _stop_early(valid_ppls, args.early_stopping, save_dir):
                break

        test_ppl, _ = run_epoch(sess, mtest, 0, iter(test_reader))
        print('Test perplexity: {:.3f}'.format(test_ppl))


if __name__ == '__main__':
    main()
