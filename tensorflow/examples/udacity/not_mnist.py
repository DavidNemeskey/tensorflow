#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Downloads and filters notMNIST."""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
from argparse import ArgumentParser
import os
import tarfile
import sys

import numpy as np
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


def parse_arguments():
    parser = ArgumentParser(
        description='Exercises for assignment 4 -- convolutions.')
    parser.add_argument('--data-dir', '-d', required=True,
                        help='directory to save the data to.')
    parser.add_argument('--train-size', type=int, default=-1,
                        help='the size of the training set [-1]. A negative '
                             'number indicates that all data is needed.')
    parser.add_argument('--valid-size', type=int, default=20000,
                        help='the size of the validation set [20(k)].')
    parser.add_argument('--test-size', type=int, default=-1,
                        help='the size of the test set [-1]. A negative '
                             'number indicates that all data is needed.')
    parser.add_argument('--filter-duplicates', '-f', action='store_true',
                        help='whether to filter duplicates or not.')
    args = parser.parse_args()

    if args.valid_size <= 0:
        parser.error('--valid-size must be greater than 0.')
    return (args.data_dir, args.train_size, args.valid_size, args.test_size,
            args.filter_duplicates)


def download_progress_hook(count, blockSize, totalSize):
    """
    A hook to report the progress of a download. This is mostly intended for users
    with slow internet connections. Reports every 1% change in download progress.
    """
    last_percent_reported = None
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(url, data_dir, filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    real_fn = os.path.join(data_dir, filename)
    if force or not os.path.exists(real_fn):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, real_fn,
                                  reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(real_fn)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception('Failed to verify ' + url + filename +
                        '. Can you get to it with a browser?')
    return real_fn


def maybe_extract(filename, num_classes, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall(path=os.path.dirname(filename))
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders


def load_letter(folder, min_num_images, image_size=28, pixel_depth=255.0):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(image_size, pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valids, trains = [], []
    # TODO lossless
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_size != 0:
                    valid_dataset, valid_labels = make_arrays(
                        vsize_per_class, image_size)
                    valid_dataset[:, :, :] = letter_set[:vsize_per_class, :, :]
                    valid_labels[:] = label
                    valids.append((valid_dataset, valid_labels))
                train_letter = letter_set[vsize_per_class:, :, :]
                if train_size < 0:
                    tsize_per_class = train_letter.shape[0]
                train_dataset, train_labels = make_arrays(
                    tsize_per_class, image_size)
                train_dataset[:, :, :] = train_letter[:tsize_per_class, :, :]
                train_labels[:] = label
                trains.append((train_dataset, train_labels))
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    train_dataset, train_labels = map(np.concatenate, zip(*trains))
    if valid_size != 0:
        valid_dataset, valid_labels = map(np.concatenate, zip(*valids))
    else:
        valid_dataset, valid_labels = None, None

    return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def save_data(data_dir, train_dataset, train_labels, valid_dataset,
              valid_labels, test_dataset, test_labels,
              pickle_file='notMNIST.pickle'):
    pickle_path = os.path.join(data_dir, pickle_file)
    try:
        f = open(pickle_path, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def main():
    (data_dir, train_size, valid_size,
     test_size, filter_duplicates) = parse_arguments()

    image_size = 28

    # Download data
    url = 'http://commondatastorage.googleapis.com/books1000/'
    train_filename = maybe_download(url, data_dir, 'notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download(url, data_dir, 'notMNIST_small.tar.gz', 8458043)

    # Extract it from the gz files
    num_classes = 10
    np.random.seed(42)
    train_folders = maybe_extract(train_filename, num_classes)
    test_folders = maybe_extract(test_filename, num_classes)

    # Pickle it letter by letter
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)

    # Merge and prune the datasets
    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
        image_size, train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(
        image_size, test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    # Randomize the data
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

    if filter_duplicates:
        pass

    # Save the bitch!
    file_name = 'notMNIST{}_{}_{}_{}.pickle'.format(
        '_filtered' if filter_duplicates else '', train_dataset.shape[0],
        valid_dataset.shape[0], test_dataset.shape[0])
    save_data(data_dir, train_dataset, train_labels, valid_dataset, valid_labels,
              test_dataset, test_labels, file_name)


def load_data(data_dir='/home/david/Research/udacity/tensorflow',
              pickle_file='notMNIST.pickle'):
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


if __name__ == '__main__':
    main()
