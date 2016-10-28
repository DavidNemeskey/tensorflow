#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Prepares data for training."""

from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
import subprocess

import numpy as np

from auxiliary import openall
from data_input import digits_format_str


class CorpusCompiler(object):
    def __init__(self, input_files, output_prefix):
        self.input_files = input_files
        self.output_prefix = output_prefix
        self.input_size = self.input_length(input_files)

    @staticmethod
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

    @staticmethod
    def read_input(input_files):
        """Reads the input files one-by-one and yields tokens line-by-line."""
        for input_file in input_files:
            with openall(input_file) as inf:
                for line in inf:
                    yield line.strip().split() + ['</s>']


class TxtDiskCompiler(CorpusCompiler):
    def __init__(self, input_files, output_prefix, batch_size):
        super(TxtDiskCompiler, self).__init__(input_files, output_prefix)
        self.batch_size = batch_size

    def __call__(self):
        """Does the work."""
        num_outs = self.batch_size  # The modulo is dropped
        out_size = self.input_size // num_outs
        out_ext = digits_format_str(num_outs)

        with openall(self.output_prefix, 'wt') as header:
            print('TXT_DISK\t{}\t{}'.format(
                num_outs, self.input_size), file=header)

        input_iter = self.read_input(self.input_files)
        tokens = []
        for outi in range(num_outs):
            written = 0
            with openall(self.output_prefix + out_ext.format(outi), 'wt') as outf:
                while written < out_size:
                    tokens.extend(next(input_iter))
                    tokens_to_write = tokens[:out_size - written]
                    print(('\n'.join(tokens_to_write)), file=outf)
                    written += len(tokens_to_write)
                    tokens = tokens[len(tokens_to_write):]


class IntMemCompiler(CorpusCompiler):
    def __init__(self, input_files, output_prefix, vocab_file):
        super(IntMemCompiler, self).__init__(input_files, output_prefix)
        self.read_vocab(vocab_file)

    def read_vocab(self, vocab_file):
        with openall(vocab_file) as inf:
            self.vocab = {w: i for i, w in enumerate(l.strip().split('\t')[0]
                          for l in inf)}

    def __call__(self):
        """Does the work."""
        with openall(self.output_prefix, 'wt') as header:
            print('INT_MEM\t{}'.format(self.input_size), file=header)

        arr = np.zeros((self.input_size), dtype=np.int32)
        index = 0
        for tokens in self.read_input(self.input_files):
            for token in tokens:
                arr[index] = self.vocab[token]
                index += 1

        np.savez(self.output_prefix, data=arr)


def parse_arguments():
    parser = ArgumentParser(
        description='Prepares data for training.')
    parser.add_argument('input_files', nargs='+',
                        help='the tokenized input text files.')
    parser.add_argument('--output-prefix', '-o', required=True,
                        help='the prefix of the output files\' names.')
    parser.add_argument('--format', '-f', choices=['txt_disk', 'int_mem'],
                        help='the data format. The two choices are txt_disk, '
                             'where the data is on disk in tokenized text '
                             'format, and int_mem, where the data is in '
                             'memory as an array of ints.')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='the training batch size. Only valid for the '
                             'txt_disk format.')
    parser.add_argument('--vocab-file', '-v', default=None,
                        help='the vocabulary file, created by count_vocab.py. '
                             'Only needed for the int_mem format.')
    args = parser.parse_args()

    if args.format == 'txt_disk' and not args.batch_size:
        parser.error('The number of batches is a required argument of the '
                     'txt_disk format.')
    if args.format == 'int_mem' and not args.vocab_file:
        parser.error('The vocabulary file is a required argument of the '
                     'int_mem format.')

    return args


def main():
    args = parse_arguments()
    if args.format == 'txt_disk':
        c = TxtDiskCompiler(args.input_files, args.output_prefix, args.batch_size)
    else:
        c = IntMemCompiler(args.input_files, args.output_prefix, args.vocab_file)
    c()


if __name__ == '__main__':
    main()
