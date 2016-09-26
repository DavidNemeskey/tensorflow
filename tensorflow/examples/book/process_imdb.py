#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Preprocesses the IMDB sentiment dataset."""
from __future__ import absolute_import, division, print_function
from argparse import ArgumentParser
from io import open
import re
import tarfile


TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')


def parse_arguments():
    parser = ArgumentParser(
        description='Preprocesses the IMDB sentiment dataset.')
    parser.add_argument('input_file', help='the IMDB data (.tar) file.')
    parser.add_argument('output_prefix',
                        help='the prefix of the output file. Two files are '
                             'written: a vocabulary file (.vocab) and the '
                             'output file (.tsv).')
    args = parser.parse_args()
    return args.input_file, args.output_prefix


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


def main():
    input_file, output_prefix = parse_arguments()
    vocab = set()
    with open(output_prefix + '.tsv', 'wt', encoding='utf-8') as outf:
        for text, judgement in read_tarfile(input_file):
            vocab.update(text)
            print('{}\t{}'.format(judgement, ' '.join(text)), file=outf)
    with open(output_prefix + '.vocab', 'wt', encoding='utf-8') as outf:
        for word in sorted(vocab):
            print(word, file=outf)


if __name__ == '__main__':
    main()
