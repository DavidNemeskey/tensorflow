#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Preparations common to all exercises."""

import os

import numpy as np
from six.moves import cPickle as pickle


def labels_to_one_hot(labels, num_labels=10):
    """Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]."""
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return labels


def reformat_1d(dataset, labels, image_size=28, num_labels=10):
    """
    Reformats the image data to a one dimensional tensor (two with the batch
    / dataset size included).
    """
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    return dataset, labels_to_one_hot(labels, num_labels)


def reformat_conv(dataset, labels, image_size=28, num_labels=10, num_channels=1):
    """
    Reformats the image data to a three dimensional tensor (four with the batch
    / dataset size included): height, width, depth=number of channels.
    """
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset, labels_to_one_hot(labels, num_labels)


def accuracy(predictions, labels):
    """Accuracy on one-hot representations."""
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


if __name__ == '__main__':
    load_data()
