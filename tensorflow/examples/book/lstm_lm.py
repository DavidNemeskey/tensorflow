#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Generic language modeling with RNN."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
import glob
import os
import re
import time

import numpy as np
import tensorflow as tf

from auxiliary import AttrDict
from data_input import data_loader
from lstm_model import LSTMModel
from rnn import get_cell_types
from softmax import get_loss_function

TEST_BATCH, TEST_STEPS = 1, 1


def get_sconfig(gpu_memory):
    """
    Returns a session configuration object that sets the GPU memory limit.
    """
    params = {}
    # params = {'log_device_placement': True}
    if gpu_memory:
        params['gpu_options'] = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory)
    else:
        return tf.ConfigProto(**params)


def parse_arguments():
    parser = ArgumentParser(
        description='Character-based language modeling with RNN.')
    parser.add_argument('train_file', help='the text file to train on.')
    parser.add_argument('valid_file',
                        help='the text file to use as a validation set.')
    parser.add_argument('test_file',
                        help='the text file to use as a test set.')
    parser.add_argument('vocab_file',
                        help='the vocabulary file.')
    parser.add_argument('--model-name', '-m', default='RNN CLM',
                        help='the name of the model [RNN CLM].')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='the training batch size [100].')
    parser.add_argument('--num-nodes', '-N', type=int, default=200,
                        help='use how many RNN cells [200].')
    parser.add_argument('--num-steps', '-s', type=int, default=20,
                        help='how many steps to unroll the network for [20].')
    parser.add_argument('--rnn-cell', '-C', choices=get_cell_types().keys(),
                        default='lstm', help='the RNN cell to use [lstm].')
    parser.add_argument('--layers', '-L', type=int, default=1,
                        help='the number of RNN laercell to use [lstm].')
    parser.add_argument('--dropout', '-D', type=float, default=1.0,
                        help='the keep probability of dropout; if not ' +
                             'specified, no dropout is applied.')
    parser.add_argument('--embedding', '-E', choices={'no', 'yes'}, default='yes',
                        help='whether to compute an embedding as well [yes].')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='the default number of epochs [20].')
    # parser.add_argument('--epoch-size', '-s', type=int, default=0,
    #                     help='the default epoch size. The number of batches '
    #                          'processed in an epoch. If 0 (the default), '
    #                          'the whole data is processed in an apoch.')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.02,
                        help='the default learning rate [0.02].')
    parser.add_argument('--lr-decay', '-d', type=float, default=0.5,
                        help='the learning rate decay [0.5].')
    parser.add_argument('--decay-delay', type=float, default=None,
                        help='keep the learning rate constant for this many '
                             'epochs [learning_rate // 4 + 1].')
    parser.add_argument('--init-scale', '-i', type=float, default=0.1,
                        help='the initial scale of the weights [0.1].')
    parser.add_argument('--max-grad-norm', '-g', type=float, default=None,
                        help='the limit for gradient clipping [None].')
    parser.add_argument('--verbose', '-v', type=int, default=10,
                        help='print the perplexity how many times in an '
                             'epoch [10].')
    parser.add_argument('--early-stopping', type=int, default=0,
                        help='early stop after the perplexity has been '
                             'detoriating after this many steps. If 0 (the '
                             'default), do not stop early.')
    parser.add_argument('--test-only', '-T', action='store_true',
                        help='do not train, only run the evaluation. Not '
                             'really meaningful if there are no checkpoints.')
    parser.add_argument('--gpu-memory', type=float, default=None,
                        help='limit on the GPU memory ratio [None].')
    parser.add_argument('--trainsm', default='Softmax',
                        help='the softmax loss alterative to use.')
    parser.add_argument('--validsm', default='Softmax',
                        help='the softmax loss alterative to use.')
    parser.add_argument('--testsm', default='Softmax',
                        help='the softmax loss alterative to use.')
    args = parser.parse_args()

    if args.decay_delay is None:
        args.decay_delay = args.epochs // 4 + 1

    return args


def run_epoch(session, model, data, epoch_size=0, verbose=0,
              global_step=0, writer=None):
    """
    Runs an epoch on the network.
    - epoch_size: if 0, it is taken from data
    - data: a DataLoader instance
    """
    # TODO: these two should work together better, i.e. keep the previous
    #       iteration state intact if epoch_size != 0; also loop around
    epoch_size = data.epoch_size if epoch_size <= 0 else epoch_size
    data_iter = iter(data)
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = [model.cost, model.final_state, model.train_op]
    fetches_summary = fetches + [model.summaries]
    if verbose:
        log_every = epoch_size // verbose

    for step in range(epoch_size):
        x, y = next(data_iter)

        # feed_dict = {model.sequence: batch}
        # for i, (c, h) in enumerate(model.initial_state):
        #     feed_dict[c] = state[i].c  # CEC for layer i
        #     feed_dict[h] = state[i].h  # hidden for layer i
        feed_dict = {
            model.input_data: x,
            model.targets: y,
            model.initial_state: state
        }

        if verbose and step % log_every == log_every - 1:
            cost, state, _, summary = session.run(fetches_summary, feed_dict)
            if writer:
                writer.add_summary(summary, global_step=global_step)
            if model.is_training:
                global_step += 1
        else:
            cost, state, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.params.num_steps
        if verbose and step % log_every == log_every - 1:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.params.batch_size / (time.time() - start_time)))

    # global_step is what the user sees, i.e. if the output is verbose, it is
    # increased, otherwise it isn't
    if not verbose and model.is_training:
        global_step += 1

    return np.exp(costs / iters), global_step


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

    test_batch = TEST_BATCH if not args.test_only else args.batch_size
    test_steps = TEST_STEPS

    if not args.test_only:
        train_data = data_loader(args.train_file, args.batch_size, args.num_steps,
                                 vocab_file=args.vocab_file)
        valid_data = data_loader(args.valid_file, args.batch_size, args.num_steps,
                                 vocab_file=args.vocab_file)
    test_data = data_loader(args.test_file, test_batch, test_steps,
                            vocab_file=args.vocab_file)

    train_params = AttrDict(
        rnn_cell=args.rnn_cell,
        hidden_size=args.num_nodes,
        num_layers=args.layers,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        keep_prob=args.dropout,
        vocab_size=len(test_data.vocab),
        max_grad_norm=args.max_grad_norm,
        embedding=args.embedding,
        data_type=tf.float32,
    )
    train_params.batch_size = train_data.batch_size
    valid_params = AttrDict(train_params)
    valid_params.batch_size = valid_data.batch_size
    eval_params = AttrDict(train_params)
    eval_params.batch_size = test_data.batch_size
    eval_params.num_steps = test_steps

    if not args.test_only:
        trainsm = get_loss_function(
            args.trainsm, train_params.hidden_size, train_params.vocab_size,
            train_params.batch_size, train_params.num_steps, train_params.data_type)
        validsm = get_loss_function(
            args.validsm, valid_params.hidden_size, valid_params.vocab_size,
            valid_params.batch_size, valid_params.num_steps, valid_params.data_type)
    testsm = get_loss_function(
        args.testsm, eval_params.hidden_size, eval_params.vocab_size,
        eval_params.batch_size, eval_params.num_steps, eval_params.data_type)

    with tf.Graph().as_default() as graph:
        # init_scale = 1 / math.sqrt(args.num_nodes)
        initializer = tf.random_uniform_initializer(
            -args.init_scale, args.init_scale)

        if not args.test_only:
            with tf.name_scope('Train'):
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    mtrain = LSTMModel(train_params, is_training=True, softmax=trainsm)
            with tf.name_scope('Valid'):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    mvalid = LSTMModel(valid_params, is_training=False, softmax=validsm)
        with tf.name_scope('Test'):
            with tf.variable_scope("Model", reuse=not args.test_only,
                                   initializer=initializer):
                mtest = LSTMModel(eval_params, is_training=False, softmax=testsm)
        with tf.name_scope('Global_ops'):
            saver = tf.train.Saver(
                name='saver', max_to_keep=max(10, args.early_stopping + 1))
            init = tf.initialize_all_variables()

    # TODO: look into Supervisor
    # The training itself
    with tf.Session(graph=graph, config=get_sconfig(args.gpu_memory)) as sess:
        save_dir = os.path.join('saves', args.model_name)
        boards_dir = os.path.join('boards', args.model_name)
        writer = tf.train.SummaryWriter(boards_dir, graph=graph)
        last_epoch = init_or_load_session(sess, save_dir, saver, init)
        global_step = 0
        if not args.test_only:
            print('Epoch {:2d}-                 valid PPL {:6.3f}'.format(
                last_epoch, run_epoch(sess, mvalid, valid_data, 0)[0]))

            valid_ppls = []
            for epoch in range(last_epoch, args.epochs + 1):
                lr_decay = args.lr_decay ** max(epoch - args.decay_delay, 0.0)
                mtrain.assign_lr(sess, args.learning_rate * lr_decay)

                train_perplexity, global_step = run_epoch(
                    sess, mtrain, train_data, 0, verbose=args.verbose,
                    global_step=global_step, writer=writer)
                valid_perplexity, _ = run_epoch(sess, mvalid, valid_data)
                print('Epoch {:2d} train PPL {:6.3f} valid PPL {:6.3f}'.format(
                    epoch, train_perplexity, valid_perplexity))
                saver.save(sess, os.path.join(save_dir, 'model'), epoch)

                valid_ppls.append(valid_perplexity)
                # Check for overfitting
                if stop_early(valid_ppls, args.early_stopping, save_dir):
                    break

        print('Running evaluation...')
        test_perplexity, _ = run_epoch(sess, mtest, test_data)
        print('Test perplexity: {:.3f}'.format(test_perplexity))

        writer.close()


if __name__ == '__main__':
    main()
