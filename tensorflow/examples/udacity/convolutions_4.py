#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Exercises for assignment 4 -- convolutions."""
from collections import namedtuple
import tensorflow as tf

from common import accuracy, load_data, reformat_conv


def create_graph_0(graph, params):
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
            conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
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


def train_graph(graph, data, params):
    loss = graph.get_tensor_by_name('loss:0')
    prediction = graph.get_tensor_by_name('prediction:0')
    with graph.as_default():
        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        init = tf.initialize_all_variables()

    num_steps = 1001

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


def main():
    raw_data, data = load_data(), {}
    for prefix in ['train', 'valid', 'test']:
        reformat(raw_data, data, prefix)
    params = namedtuple(
        'Params', ['image_size', 'num_channels', 'num_labels',
                   'batch_size', 'patch_size', 'depth', 'num_hidden']
    )(28, 1, 10, 16, 5, 16, 64)

    graph = tf.Graph()
    create_graph_0(graph, params)
    session = train_graph(graph, data, params)
    test_model(session, data)


if __name__ == '__main__':
    main()
