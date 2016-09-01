#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Downloads and filters the data for the text assignments."""

from __future__ import print_function
from argparse import ArgumentParser
import os
import zipfile

import tensorflow as tf
from six.moves.urllib.request import urlretrieve


def parse_arguments():
    parser = ArgumentParser(
        description='Exercises for assignment 4 -- convolutions.')
    parser.add_argument('--data-dir', '-d', required=True,
                        help='directory to save the data to.')
    args = parser.parse_args()

    return args.data_dir


def maybe_download(url, data_dir, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        filename, _ = urlretrieve(url + filename, filepath)
    statinfo = os.stat(filepath)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + url + filename +
                        '. Can you get to it with a browser?')
    return filepath


def read_as_list(zip_file_name):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(zip_file_name) as f:
        return tf.compat.as_str(f.read(f.namelist()[0])).split()


def read_as_string(zip_file_name):
    """Extract the first file enclosed in a zip file as a string."""
    # WTF is this code? Lucky us, there is only one file in it...
    with zipfile.ZipFile(zip_file_name) as f:
        for name in f.namelist():
            return tf.compat.as_str(f.read(name))


def main():
    data_dir = parse_arguments()
    url = 'http://mattmahoney.net/dc/'

    data_file = maybe_download(url, data_dir, 'text8.zip', 31344016)
    print('Downloaded {}'.format(data_file))


if __name__ == '__main__':
    main()
