#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Cuts the input (files) into batch_size consecutive chunks; these include
the EoS </s> token.
"""

from argparse import ArgumentParser
import subprocess

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
    args = parser.parse_args()


def input_length(input_files):
    s = subprocess.check_output(['wc', '-wl'] + args.input_files)
    fields = s.strip().rsplit('\n', 1)[1].strip().split()
    # field 3 is 'total' if |input_files| > 1 else the name of the file
    return int(fields[0]) + int(fields[1])


def read_input(input_files):
    vocab = {}
    for input_file in input_files:
        with openall(input_file) as inf:
            for line in inf:
                tokens = line.strip().split() + ['</s>']


def main():
    args = parse_arguments()
    input_size = input_length(args.input_files)


if __name__ == '__main__':
    main()
