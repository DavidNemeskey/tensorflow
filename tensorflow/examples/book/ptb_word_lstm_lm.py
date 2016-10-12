# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
import glob
import os
import re
import time

import numpy as np
import tensorflow as tf

# from tensorflow.models.rnn.ptb import reader
from lstm_model import LSTMModel
import ptb_reader as reader

logging = tf.logging


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, data, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // model.params.batch_size) - 1) // model.params.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = model.initial_state.eval(session=session)

    fetches = [model.cost, model.final_state, model.train_op]

    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.params.batch_size,
                                                      model.params.num_steps)):
        feed_dict = {
            model.input_data: x,
            model.targets: y,
            model.initial_state: state
        }
        # print(np.vectorize(lambda e, d: d[e])(x, reader.id_to_word))
        cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.params.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.params.batch_size / (time.time() - start_time)))
    end_time = time.time()

    return np.exp(costs / iters), end_time - start_time


def get_params(args):
    if args.model == "small":
        params = SmallConfig()
    elif args.model == "medium":
        params = MediumConfig()
    elif args.model == "large":
        params = LargeConfig()
    elif args.model == "test":
        params = TestConfig()
    else:
        raise ValueError("Invalid model: %s", args.model)
    params.data_type = tf.float16 if args.fp16 else tf.float32
    params.embedding = 'yes'
    return params


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
        description='Runs the PTB LM as described in Zaremba et al. (2015).')
    parser.add_argument('data_path', help='the directory with the text files.')
    parser.add_argument('--evaluation-only', '-E', action='store_true',
                        help='do not train, only run the evaluation. Not '
                             'really meaningful if there are no checkpoints.')
    parser.add_argument('--model-name', '-n', default='PTB',
                        help='the name of the model [PTB].')
    parser.add_argument('--model', '-m', choices=['small', 'medium', 'large'],
                        default='small', help='the size of the network [small].')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='early stop after the perplexity has been '
                             'detoriating after this many steps. If 0 (the '
                             'default), do not do early stopping.')
    parser.add_argument('--gpu-memory', type=float, default=None,
                        help='limit on the GPU memory ratio [None].')
    parser.add_argument('--fp16', action='store_true',
                        help='use 16 bit floats instead of 32 bit[False].')
    args = parser.parse_args()

    return args


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


def stop_early(valid_ppls, early_stop, save_dir):
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


def main():
    args = parse_arguments()

    # TODO
    raw_data = reader.ptb_raw_data(args.data_path)
    train_data, valid_data, test_data, _ = raw_data

    params = get_params(args)
    eval_params = get_params(args)
    eval_params.batch_size = 1
    eval_params.num_steps = 1

    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-params.init_scale,
                                                    params.init_scale)

        with tf.name_scope('Train'):
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                mtrain = LSTMModel(is_training=True, params=params)
        with tf.name_scope('Valid'):
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mvalid = LSTMModel(is_training=False, params=params)
        with tf.name_scope('Test'):
            with tf.variable_scope("model", reuse=True, initializer=initializer):
                mtest = LSTMModel(is_training=False, params=eval_params)
        with tf.name_scope('Global_ops'):
            saver = tf.train.Saver(
                name='saver', max_to_keep=max(10, args.early_stopping + 1))
            init = tf.initialize_all_variables()

    # TODO: look into Supervisor
    # The training itself
    valid_ppls = []
    with tf.Session(graph=graph, config=get_sconfig(0.5)) as session:
        save_dir = os.path.join('saves', args.model_name)
        last_epoch = init_or_load_session(session, save_dir, saver, init)
        if not args.evaluation_only:
            for epoch in range(last_epoch, params.max_max_epoch):
                lr_decay = params.lr_decay ** max(epoch - params.max_epoch, 0.0)
                mtrain.assign_lr(session, params.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (epoch + 1, session.run(mtrain.lr)))
                train_perplexity, _ = run_epoch(session, mtrain, train_data,
                                                verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (epoch + 1, train_perplexity))
                valid_perplexity, _ = run_epoch(session, mvalid, valid_data)
                print("Epoch: %d Valid Perplexity: %.3f" % (epoch + 1, valid_perplexity))
                saver.save(session, os.path.join(save_dir, 'model'), epoch)

                valid_ppls.append(valid_perplexity)
                # Check for overfitting
                if stop_early(valid_ppls, args.early_stopping, save_dir):
                    break

        print('Running evaluation...')
        test_perplexity, _ = run_epoch(session, mtest, test_data)
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()
    # tf.app.run()
