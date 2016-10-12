#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Cuts the input (files) into batch_size consecutive chunks; these include
the EoS </s> token.
"""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from builtins import range
from collections import Counter
import math
import os
import subprocess
import sys

import numpy as np

from auxiliary import openall


def parse_arguments():
    parser = ArgumentParser(
        description='Cuts the input files into batch_size chunks.')
    parser.add_argument('input_files', nargs='+',
                        help='the tokenized input text files.')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='the training batch size [100].')
    parser.add_argument('--output-prefix', '-o', required=True,
                        help='the prefix of the output files\' names.')
    parser.add_argument('--vocab-file', '-v', default=None,
                        help='if specified, the vocabulary file here is '
                             'reused in favor of a dataset-specific file.')
    return parser.parse_args()


def read_vocab(vocab_file):
    if os.path.isfile(vocab_file):
        with openall(vocab_file) as inf:
            return Counter((token, int(count)) for token, count in
                           [l.split('\t') for l in inf.read().strip().split('\n')])
    else:
        return Counter()


def write_vocab(vocab, vocab_file):
    """Destructively modifies vocab (removes <unk> and </s>)."""
    with openall(vocab_file, 'wt') as outf:
        print('<unk>\t{}\n</s>\t{}'.format(
            vocab.pop('<unk>'), vocab.pop('</s>')), file=outf)
        for token, freq in vocab.most_common():
            print('{}\t{}'.format(token, freq), file=outf)


def input_length(input_files):
    s = subprocess.getoutput('wc -wl ' + ' '.join(input_files))
    fields = s.strip().rsplit('\n', 1)[-1].strip().split()
    # field 3 is 'total' if |input_files| > 1 else the name of the file
    return int(fields[0]) + int(fields[1])


def read_input(input_files, vocab):
    """Reads the input files one-by-one and yields tokens line-by-line."""
    for input_file in input_files:
        with openall(input_file) as inf:
            for line in inf:
                tokens = line.strip().split() + ['</s>']
                vocab.update(tokens)
                yield tokens


def _digits_format_str(number):
    return '.{{:0{}}}.gz'.format(int(math.ceil(math.log10(number))))


def divide_text(input_size, input_iter, batch_size, output_prefix):
    """Does the work."""
    num_outs = batch_size  # The modulo is dropped
    out_size = input_size // num_outs
    out_ext = _digits_format_str(num_outs)

    with openall(output_prefix, 'wt') as header:
        print('{}\t{}'.format(num_outs, input_size), file=header)
    tokens = []
    for outi in range(num_outs):
        written = 0
        with openall(output_prefix + out_ext.format(outi), 'wt') as outf:
            while written < out_size:
                tokens.extend(next(input_iter))
                tokens_to_write = tokens[:out_size - written]
                print(('\n'.join(tokens_to_write)), file=outf)
                written += len(tokens_to_write)
                tokens = tokens[len(tokens_to_write):]


class DataLoader(object):
    """Loads the data written by this script."""
    def __init__(self, header, batch_size, num_steps, one_hot=False,
                 data_type=np.int32, vocab=None):
        self.header = header
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.one_hot = one_hot
        self.data_type = data_type
        data_batches, self.data_len = self._read_header()
        self.queues = self._setup_queues(data_batches)
        self.vocab = self._read_vocab() if not vocab else vocab
        self.epoch_size = (
            ((self.data_len // data_batches - 1) // num_steps) *
            len(self.queues[0])
        )  # -1 because targets are shifted right by 1 step

    def __iter__(self):
        for q_step in range(len(self.queues[0])):
            infs = [openall(self.queues[i][q_step]) for i in range(self.batch_size)]
            arr = np.zeros((self.batch_size, self.num_steps + 1),
                           dtype=self.data_type)
            arr[:, -1:] = np.array(self._read_from_infs(infs, 1))
            for i in range(self.epoch_size):
                arr[:, 0] = arr[:, -1]
                arr[:, 1:] = np.array(
                    self._read_from_infs(infs, self.num_steps))
                if self.one_hot:
                    ret = np.zeros((self.batch_size, self.num_steps, len(self.vocab)),
                                   dtype=self.data_type)
                    ret[list(np.indices(ret.shape[:-1])) + [arr]] = 1
                    # for i in range(ret.shape[0]):
                    #     for j in range(ret.shape[1]):
                    #         ret[i, j, arr[i, j]] = 1
                else:
                    ret = arr
                yield ret[:, :self.num_steps], ret[:, 1:]
            for inf in infs:
                inf.close()

    def _read_from_infs(self, infs, num_tokens):
        return [[self.vocab[inf.readline().strip()] for _ in range(num_tokens)]
                for inf in infs]

    def _setup_queues(self, data_batches):
        div, mod = divmod(data_batches, self.batch_size)
        if div == 0:
            raise ValueError('Not enough batch files ({} instead of {})'.format(
                data_batches, self.batch_size))
        elif mod != 0:
            print('The number of data files ({}) '.format(data_batches) +
                  'is not compatible with the batch size ' +
                  '({}). Only using the first '.format(self.batch_size) +
                  '{} files.'.format(self.batch_size * div), file=sys.stderr)

        ext_str = _digits_format_str(data_batches)
        queues = [[] for _ in range(self.batch_size)]
        for i in range(div * self.batch_size):
            queues[i % self.batch_size].append(self.header + ext_str.format(i))
        return queues

    def _read_vocab(self):
        with openall(self.header + '.vocab.gz') as inf:
            return {token: i for i, token in
                    enumerate(inf.read().strip().split())}

    def _read_header(self):
        with openall(self.header) as inf:
            data_batches, data_len = inf.readline().strip().split('\t')
        return int(data_batches), int(data_len)


def main():
    args = parse_arguments()
    vocab = read_vocab(args.vocab_file) if args.vocab_file else Counter()
    input_size = input_length(args.input_files)
    input_iter = read_input(args.input_files, vocab)
    divide_text(input_size, input_iter, args.batch_size, args.output_prefix)
    write_vocab(vocab, args.vocab_file if args.vocab_file
                                       else args.output_prefix + '.vocab.gz')


if __name__ == '__main__':
    main()
