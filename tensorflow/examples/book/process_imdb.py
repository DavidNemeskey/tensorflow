#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Preprocesses the IMDB sentiment dataset."""
from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from io import open
import random
import re
import tarfile


TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')


def parse_arguments():
    parser = ArgumentParser(
        description='Preprocesses the IMDB sentiment dataset.')
    parser.add_argument('input_file', help='the IMDB data (.tar) file.')
    parser.add_argument('output_prefix',
                        help='the prefix of the output file. Four files are '
                             'written: a vocabulary file (.vocab) and the '
                             'output files ({train,valid,test}.tsv).')
    parser.add_argument('--valid-size', '-v', type=int, default=1000,
                        help='the validation set size.')
    parser.add_argument('--test-size', '-t', type=int, default=1000,
                        help='the test set size.')
    parser.add_argument('--random-seed', '-r', type=int, default=42,
                        help='the random seed for shuffling the data.')
    args = parser.parse_args()
    return (args.input_file, args.output_prefix, args.valid_size,
            args.test_size, args.random_seed)


def read_tarfile(fn):
    with tarfile.open(fn) as archive:
        for filename in archive.getnames():
            if filename.startswith('aclImdb/train/pos/'):
                yield _read(archive, filename), True
            elif filename.startswith('aclImdb/train/neg/'):
                yield _read(archive, filename), False


def _read(archive, filename):
    with archive.extractfile(filename) as f:
        return [x.lower() for x in TOKEN_REGEX.findall(f.read().decode('utf-8'))]


def write_file(fn, data):
    with open(fn, 'wt', encoding='utf-8') as outf:
        for text, judgement in data:
            print('{}\t{}'.format(judgement, ' '.join(text)), file=outf)


def main():
    input_file, output_prefix, valid_size, test_size, seed = parse_arguments()
    data = list(read_tarfile(input_file))

    # Write the vocabulary file
    vocab = set()
    for text, _ in data:
        vocab.update(text)
    with open(output_prefix + '.vocab', 'wt', encoding='utf-8') as outf:
        for word in sorted(vocab):
            print(word, file=outf)

    random.seed(seed)
    random.shuffle(data)
    write_file(output_prefix + '.train.tsv', data[:-(valid_size + test_size)])
    write_file(output_prefix + '.valid.tsv', data[-(valid_size + test_size):-test_size])
    write_file(output_prefix + '.test.tsv', data[-test_size:])


if __name__ == '__main__':
    main()
