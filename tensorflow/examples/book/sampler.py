#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Samples from the character-based LM."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
from functools import partial

import numpy as np
import tensorflow as tf

from arxiv_fetcher import Preprocessing
from auxiliary import AttrDict
from lstm_clm import CharacterLM


def parse_arguments():
    parser = ArgumentParser(
        description='Character-based language modeling with RNN.')
    parser.add_argument('--model-name', '-m', default='RNN CLM',
                        help='the name of the model [RNN CLM].')
    parser.add_argument('--num-nodes', '-n', type=int, default=200,
                        help='use how many RNN cells [200].')
    parser.add_argument('--rnn-cell', '-c', choices=['rnn', 'lstm', 'gru'],
                        default='lstm', help='the RNN cell to use [lstm].')
    parser.add_argument('--layers', '-L', type=int, default=1,
                        help='the number of RNN laercell to use [lstm].')
    parser.add_argument('--sampling-temperature', '-s', type=float, default=0.5,
                        help='sampling temperature ("adventuroussness").')
    parser.add_argument('--text-length', '-l', type=int, default=100,
                        help='length of the generated text.')
    args = parser.parse_args()

    if args.rnn_cell == 'gru':
        args.rnn_cell = tf.nn.rnn_cell.GRUCell
    elif args.rnn_cell == 'lstm':
        args.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
    else:
        args.rnn_cell = tf.nn.rnn_cell.BasicRnnCell
    return args


def sample(distribution, sampling_temperature, vocabulary):
    dist = np.log(distribution) / sampling_temperature
    dist = np.exp(dist) / np.exp(dist).sum()
    choice = np.random.choice(len(dist), p=dist)
    choice = vocabulary[choice]
    return choice


def create_state(params):
    """For injecting our state."""
    return tf.placeholder(tf.float32,
                          [1, 2 * params.rnn_hidden * params.rnn_layers])


def main():
    args = parse_arguments()
    params = AttrDict(
        name=args.model_name,
        rnn_cell=args.rnn_cell,
        rnn_hidden=args.num_nodes,
        rnn_layers=args.layers,
        batch_size=1,
        max_length=2,
        vocabulary=len(Preprocessing.VOCABULARY),
        learning_rate=0,
        gradient_clipping=None,
    )
    prep = Preprocessing([], 2, 1)
    lm = CharacterLM(params, partial(create_state, params))

    with tf.Session(graph=lm.graph) as sess:
        checkpoint = tf.train.get_checkpoint_state(lm.save_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            tf.train.Saver().restore(sess, checkpoint.model_checkpoint_path)
        else:
            print('Sampling from untrained model.')
        print('Sampling temperature', args.sampling_temperature)

        text = 'We'
        state_value = np.zeros((1, 2 * params.rnn_hidden * params.rnn_layers))
        for _ in range(args.text_length):
            feed = {lm.initial: state_value, lm.sequence: prep([text[-1] + '?'])}
            prediction, state_value = sess.run([lm.prediction, lm.state], feed)
            text += sample(prediction[0, 0], args.sampling_temperature,
                           prep.VOCABULARY)
        print(text)


if __name__ == '__main__':
    main()
