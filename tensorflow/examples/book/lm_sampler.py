#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Samples from the word-based LM."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
import os
import sys
import time

import numpy as np
import tensorflow as tf

from auxiliary import AttrDict, openall
from lstm_model import LSTMModel
from softmax import get_loss_function


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
        description='Samples from the word-based LM.')
    parser.add_argument('model_name', help='the name of the model to use.')
    parser.add_argument('vocab_file', help='the vocabulary file.')
    parser.add_argument('no_tokens', type=int,
                        help='the number of tokens to generate.')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='the training batch size [100].')
    parser.add_argument('--sampling-temperature', '-s', type=float, default=1.0,
                        help='sampling temperature ("adventurousness").')
    parser.add_argument('--num-nodes', '-n', type=int, default=200,
                        help='use how many RNN cells [200].')
    parser.add_argument('--layers', '-L', type=int, default=1,
                        help='the number of RNN laercell to use [lstm].')
    parser.add_argument('--embedding', '-E', choices={'no', 'yes'}, default='yes',
                        help='whether to compute an embedding as well [yes].')
    parser.add_argument('--gpu-memory', type=float, default=None,
                        help='limit on the GPU memory ratio [None].')
    return parser.parse_args()


def read_vocab(vocab_file):
    with openall(vocab_file) as inf:
        vocab = [l.strip().split()[0] for l in inf]
        eos = vocab.index('</s>')
        return vocab, eos


def load_session(sess, save_dir, saver):
    """Initiates or loads a session."""
    checkpoint = tf.train.get_checkpoint_state(save_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        path = checkpoint.model_checkpoint_path
        print('Loading checkpoint...', path, file=sys.stderr)
        saver.restore(sess, path)
    else:
        raise ValueError('No model found in {}'.format(save_dir))


def sample(distribution, sampling_temperature, vocabulary):
    dist = np.log(distribution) / sampling_temperature
    dist = np.exp(dist)
    dist = dist / dist.sum(axis=1)[:, np.newaxis]
    # choice = np.random.choice(len(dist), p=dist)
    return np.apply_along_axis(lambda a: np.random.choice(len(a), p=a), 1, dist)


def main():
    args = parse_arguments()
    vocab, eos = read_vocab(args.vocab_file)

    params = AttrDict(
        hidden_size=args.num_nodes,
        num_layers=args.layers,
        batch_size=args.batch_size,
        num_steps=1,
        vocab_size=len(vocab),
        embedding=args.embedding,
        data_type=tf.float32,
    )

    testsm = get_loss_function('Softmax', params.hidden_size,
                               params.vocab_size, args.batch_size, 1, params.data_type)

    with tf.Graph().as_default() as graph:
        with tf.name_scope('Test'):
            with tf.variable_scope("Model"):
                model = LSTMModel(params, is_training=False, softmax=testsm,
                                  need_prediction=True)
        with tf.name_scope('Global_ops'):
            saver = tf.train.Saver(name='saver', max_to_keep=1000)

    with tf.Session(graph=graph, config=get_sconfig(args.gpu_memory)) as sess:
        save_dir = os.path.join('saves', args.model_name)
        load_session(sess, save_dir, saver)

        fetches = [model.prediction, model.final_state]
        epochs = args.no_tokens // args.batch_size
        results = np.zeros((args.batch_size, epochs), dtype=np.int32)
        input = np.full((args.batch_size, 1), eos, dtype=np.int32)
        start_time = time.time()
        state = sess.run(model.initial_state)
        for epoch in range(epochs):
            feed_dict = {model.input_data: input,
                         model.initial_state: state}
            predictions, state = sess.run(fetches, feed_dict)
            sampled = sample(predictions.reshape(args.batch_size, params.vocab_size),
                             args.sampling_temperature, vocab)
            input = sampled.reshape(args.batch_size, 1)
            results[:, epoch] = sampled
        end_time = time.time()

        faulty_rows = len(results)
        for row in results:
            eoses = np.where(row == eos)[0]
            if len(eoses) > 0:
                print(' '.join(vocab[t] for t in row[:eoses[-1] + 1]), end=' ')
                faulty_rows -= 1
        print('Processed {} tokens in {} seconds; deleted {} rows.'.format(
            args.no_tokens, end_time - start_time, faulty_rows), file=sys.stderr)


if __name__ == '__main__':
    main()
