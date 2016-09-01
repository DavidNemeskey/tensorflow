#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Exercises for assignment 4 -- convolutions."""
from argparse import ArgumentParser
from collections import namedtuple
import os

import tensorflow as tf

from common import accuracy, reformat_conv
from not_mnist import load_data


def create_graph_01(graph, params, pooling):
    with graph.as_default():
        # Input data.
        tf_dataset = tf.placeholder(
            tf.float32, shape=(None, params.image_size, params.image_size, params.num_channels),
            name='tf_dataset')
        tf_labels = tf.placeholder(
            tf.float32, shape=(None, params.num_labels), name='tf_labels')

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [params.patch_size, params.patch_size,
             params.num_channels, params.depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([params.depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [params.patch_size, params.patch_size, params.depth, params.depth],
            stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[params.depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [params.image_size // 4 * params.image_size // 4 * params.depth,
             params.num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[params.num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [params.num_hidden, params.num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[params.num_labels]))

        # Model.
        def model(data):
            # Pooling
            def max_pool_2x2(x):
                if pooling:
                    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')
                else:
                    return x
            strides = [1, 1, 1, 1] if pooling else [1, 2, 2, 1]

            # Building the net
            conv = tf.nn.conv2d(data, layer1_weights, strides, padding='SAME')
            hidden = max_pool_2x2(tf.nn.relu(conv + layer1_biases))
            conv = tf.nn.conv2d(hidden, layer2_weights, strides, padding='SAME')
            hidden = max_pool_2x2(tf.nn.relu(conv + layer2_biases))
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(tf_dataset)
        loss = tf.reduce_mean(  # noqa
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels),
            name='loss')

        # Predictions for the training, validation, and test data.
        prediction = tf.nn.softmax(logits, name='prediction')  # noqa


def create_graph_0(graph, params):
    create_graph_01(graph, params, False)


def create_graph_1(graph, params):
    """
    The convolutional model above uses convolutions with stride 2 to reduce the
    dimensionality. Replace the strides by a max pooling operation
    (nn.max_pool()) of stride 2 and kernel size 2.
    """
    create_graph_01(graph, params, True)


def create_graph_2(graph, params):
    """
    Try to get the best performance you can using a convolutional net. Look for
    example at the classic LeNet5 architecture, adding Dropout, and/or adding
    learning rate decay.
    """
    pass


def train_graph(graph, data, params, num_steps=1001):
    loss = graph.get_tensor_by_name('loss:0')
    prediction = graph.get_tensor_by_name('prediction:0')
    with graph.as_default():
        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        init = tf.initialize_all_variables()

    session = tf.Session(graph=graph)
    with session.as_default():
        # tf.initialize_all_variables().run(session=session)
        session.run(init)
        print('Initialized')
        for step in range(num_steps):
            offset = ((step * params.batch_size) %
                      (data['trainl'].shape[0] - params.batch_size))
            batch_data = data['traind'][offset:(offset + params.batch_size), :, :, :]
            batch_labels = data['trainl'][offset:(offset + params.batch_size), :]
            feed_dict = {'tf_dataset:0': batch_data, 'tf_labels:0': batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(
                    predictions, batch_labels))
                valid_predictions = session.run(
                    prediction, feed_dict={'tf_dataset:0': data['validd']})
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_predictions, data['validl']))
        return session


def test_model(session, data):
    prediction = session.graph.get_tensor_by_name('prediction:0')
    test_predictions = session.run(
        prediction, feed_dict={'tf_dataset:0': data['testd']})
    print('Test accuracy: %.1f%%' % accuracy(
        test_predictions, data['testl']))


def reformat(raw_data, data, prefix):
    d, l = reformat_conv(raw_data['{}_dataset'.format(prefix)],
                         raw_data['{}_labels'.format(prefix)])
    data['{}d'.format(prefix)] = d
    data['{}l'.format(prefix)] = l


def parse_arguments():
    parser = ArgumentParser(
        description='Exercises for assignment 4 -- convolutions.')
    parser.add_argument('--exercise', '-e', type=int, required=True,
                        help='the exercise to run.')
    parser.add_argument('--data-file', '-f', required=True,
                        help='the pickled data file.')
    parser.add_argument('--iterations', '-i', type=int, default=1001,
                        help='the number of iterations [1001].')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='the training batch size [16].')
    parser.add_argument('--patch-size', '-p', type=int, default=5,
                        help='the patch size for the convolution [5].')
    parser.add_argument('--depth', '-d', type=int, default=16,
                        help='the depth of the convolution [16].')
    parser.add_argument('--hidden', '-H', type=int, default=64,
                        help='the size of the hidden layer[64].')
    parser.add_argument('--image-size', type=int, default=28,
                        help='the image size [28].')
    parser.add_argument('--num-channels', type=int, default=1,
                        help='the number of image channels [1].')
    args = parser.parse_args()

    return (args.exercise, args.data_file, args.iterations, args.batch_size,
            args.patch_size, args.depth, args.hidden, args.image_size,
            args.num_channels)


def main():
    (exercise, data_file, iterations, batch_size,
     patch_size, depth, hidden, image_size, num_channels) = parse_arguments()

    raw_data, data = load_data(
        os.path.dirname(data_file), os.path.basename(data_file)), {}
    for prefix in ['train', 'valid', 'test']:
        reformat(raw_data, data, prefix)
    params = namedtuple(
        'Params', ['image_size', 'num_channels', 'num_labels',
                   'batch_size', 'patch_size', 'depth', 'num_hidden']
    )(image_size, num_channels, data['trainl'].shape[1],
      batch_size, patch_size, depth, hidden)

    graph = tf.Graph()
    globals()['create_graph_{}'.format(exercise)](graph, params)
    session = train_graph(graph, data, params, iterations)
    test_model(session, data)


if __name__ == '__main__':
    main()
