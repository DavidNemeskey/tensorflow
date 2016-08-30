#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Preparations common to all exercises."""

import os

import numpy as np
from six.moves import cPickle as pickle


def load_data(data_dir='/home/david/Research/udacity/tensorflow',
              pickle_file='notMNIST_filtered.pickle'):
    """Returns the train, valid, test dataset and labels."""
    pickle_path = os.path.join(data_dir, pickle_file)

    with open(pickle_path, 'rb') as f:
        save = pickle.load(f, encoding='bytes')
        if type(next(iter(save.keys()))) == bytes:
            # Saved in Python 2
            ret = {k.decode('utf-8'): v for k, v in save.items()}
        else:
            ret = save
        print('Training set', ret['train_dataset'].shape, ret['train_labels'].shape)
        print('Validation set', ret['valid_dataset'].shape, ret['valid_labels'].shape)
        print('Test set', ret['test_dataset'].shape, ret['test_labels'].shape)

        return ret


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
