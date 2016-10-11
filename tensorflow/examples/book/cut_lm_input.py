#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Cuts the input (files) into batch_size consecutive chunks; these include
the EoS </s> token.
"""

from argparse import ArgumentParser
import math
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
    s = subprocess.check_output(['wc', '-wl'] + input_files)
    fields = s.strip().rsplit('\n', 1)[1].strip().split()
    # field 3 is 'total' if |input_files| > 1 else the name of the file
    return int(fields[0]) + int(fields[1])


def read_input(input_files, vocab):
    """Reads the input files one-by-one and yields tokens line-by-line."""
    for input_file in input_files:
        with openall(input_file) as inf:
            for line in inf:
                tokens = line.strip().split() + ['</s>']
                yield tokens


def divide_text(input_size, input_iter, batch_size, output_prefix):
    """Does the work."""
    num_outs = input_size // batch_size  # The modulo is dropped
    out_size = input_size // num_outs
    out_ext = '.{{:0{}}}'.format(int(math.ceil(math.log10(num_outs))))

    with openall(output_prefix, 'wt') as header:
        print('{}\t{}'.format(num_outs, input_size), file=header)
    tokens = []
    for outi in range(num_outs):
        written = 0
        with openall(output_prefix + out_ext.format(outi), 'wt') as outf:
            tokens.extend(next(input_iter))
            tokens_to_write = tokens[:out_size - written]
            print(('\n'.join(tokens_to_write)), file=outf)
            written += len(tokens_to_write)
            tokens = tokens[len(tokens_to_write):]
            if written == out_size:
                continue


def main():
    args = parse_arguments()
    vocab = {}
    input_size = input_length(args.input_files, vocab)


if __name__ == '__main__':
    main()
