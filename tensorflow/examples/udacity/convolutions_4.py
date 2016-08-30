#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Exercises for assignment 4 -- convolutions."""
import tensorflow as tf

from common import accuracy, load_data, reformat_conv

def exercise_0(data, image_size=28, num_labels=10, num_channels=1):
    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(data['validd'])
        tf_test_dataset = tf.constant(data['testd'])

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal(
            [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):
            conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

        num_steps = 1001

        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print('Initialized')
            for step in range(num_steps):
                offset = (step * batch_size) % (data['trainl'].shape[0] - batch_size)
                batch_data = data['traind'][offset:(offset + batch_size), :, :, :]
                batch_labels = data['trainl'][offset:(offset + batch_size), :]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                _, l, predictions = session.run(
                    [optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 50 == 0):
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy(
                        predictions, batch_labels))
                    print('Validation accuracy: %.1f%%' % accuracy(
                        valid_prediction.eval(), data['validl']))
            print('Test accuracy: %.1f%%' % accuracy(
                test_prediction.eval(), data['testl']))


def reformat(raw_data, data, prefix):
    d, l = reformat_conv(raw_data['{}_dataset'.format(prefix)],
                         raw_data['{}_labels'.format(prefix)])
    data['{}d'.format(prefix)] = d
    data['{}l'.format(prefix)] = l


def main():
    raw_data, data = load_data(), {}
    for prefix in ['train', 'valid', 'test']:
        reformat(raw_data, data, prefix)
    exercise_0(data)


if __name__ == '__main__':
    main()
