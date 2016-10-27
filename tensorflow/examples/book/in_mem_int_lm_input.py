#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Saves (and loads) the input (files) as a single file (1 line header + int32s).
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
import subprocess

import numpy as np

from auxiliary import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Converts the input files to binary input for the LM.')
    parser.add_argument('input_files', nargs='+',
                        help='the tokenized input text files.')
    parser.add_argument('--output-prefix', '-o', required=True,
                        help='the prefix of the output files\' names.')
    parser.add_argument('--vocab-file', '-v', required=True,
                        help='the vocabulary file, created by count_vocab.py.')
    return parser.parse_args()


def input_length(input_files):
    sum_len = 0
    for input_file in input_files:
        if input_file.endswith('.gz'):
            cmd = 'zcat'
        elif input_file.endswith('.bz2'):
            cmd = 'bzcat'
        else:
            cmd = 'cat'
        s = subprocess.getoutput('{} "{}" | wc -wl'.format(cmd, input_file))
        fields = s.strip().rsplit('\n', 1)[-1].strip().split()
        sum_len += int(fields[0]) + int(fields[1])
    return sum_len


def read_vocab(vocab_file):
    with openall(vocab_file) as inf:
        return {w: i for i, w in enumerate(l.strip().split('\t')[0] for l in inf)}


def read_input(input_files):
    """Reads the input files one-by-one and yields tokens line-by-line."""
    for input_file in input_files:
        with openall(input_file) as inf:
            for line in inf:
                yield line.strip().split() + ['</s>']


def convert_text(input_iter, vocab, input_size, output_prefix):
    """Does the work."""
    with openall(output_prefix, 'wt') as header:
        print('INT_MEM\t{}'.format(input_size), file=header)

    arr = np.zeros((input_size), dtype=np.int32)
    index = 0
    for tokens in input_iter:
        for token in tokens:
            arr[index] = vocab[token]
            index += 1

    assert index == input_size
    np.savez(output_prefix, data=arr)


class DataLoader(object):
    """Loads the data written by this script."""
    def __init__(self, header, batch_size, num_steps, one_hot=False,
                 data_type=np.int32, vocab_file=None):
        self.header = header
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.one_hot = one_hot
        self.data_type = data_type
        data_batches, self.data_len = batch_size, self._read_header()
        self.epoch_size = (self.data_len // data_batches - 1) // num_steps
        self.data = np.load(header + '.npz')['data'][:self.data_len // data_batches * data_batches].reshape(batch_size, -1)

    def _read_header(self):
        with openall(self.header) as inf:
            data_len = inf.readline().strip().split('\t')[1]
        return int(data_len)

    def __iter__(self):
        num_steps = self.num_steps
        for i in range(self.epoch_size):
            start = i * num_steps
            end = start + num_steps
            yield self.data[:, start:end], self.data[:, start + 1:end + 1]


def main():
    args = parse_arguments()
    vocab = read_vocab(args.vocab_file)
    input_size = input_length(args.input_files)
    input_iter = read_input(args.input_files)
    convert_text(input_iter, vocab, input_size, args.output_prefix)

if __name__ == '__main__':
    main()
