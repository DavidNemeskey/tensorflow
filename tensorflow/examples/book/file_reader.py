#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Reads tokenized text files and prepares them for use in tf."""
import numpy as np

from auxiliary import openall


class FileReader(object):
    def __init__(self, filename, vocab_map_file, batch_size, num_steps):
        self.filename = filename
        self.vocab_map = self._read_vocab_map(vocab_map_file)
        self.batch_size = batch_size
        self.num_steps = num_steps

    @staticmethod
    def _read_vocab_map(vocab_map_file):
        with openall(vocab_map_file) as inf:
            tokens = inf.read().split()
        if '</s>' not in tokens:
            tokens.append('</s>')
        return {token: i for i, token in enumerate(tokens)}

    def _convert(self, tokens):
        """Converts tokens to numbers."""
        batches = np.zeros((self.batch_size * self.num_steps, len(self.vocab_map)))
        for i, token in enumerate(tokens):
            batches[i, self.vocab_map[token]] = 1
        batches.resize((self.batch_size, self.num_steps, len(self.vocab_map)))
        return batches


class StreamFileReader(FileReader):
    """Reads a file."""
    def __init__(self, filename, vocab_map_file, batch_size, num_steps):
        super().__init__(filename, vocab_map_file, batch_size, num_steps)
        self.tokens = []
        self.len_needed = self.batch_size * self.num_steps

    def __iter__(self):
        while True:
            for _ in self._file_iterator(self.filename, True):
                if len(self.tokens) > 0:
                    yield self._convert(self.tokens[:self.len_needed])
                    self.tokens = self.tokens[min(self.len_needed, len(self.tokens)):]

    def _file_iterator(self, fn, dont_stop=False):
        while True:
            with openall(fn) as inf:
                for line in inf:
                    if len(self.tokens) < self.len_needed:
                        self.tokens.extend(line.strip().split() + ['</s>'])
                    else:
                        yield
                if dont_stop:
                    yield
                    break


class InMemoryFileReader(FileReader):
    """I am not sure we will even need this."""
    def __init__(self, filename, vocab_map_file, batch_size, num_steps):
        super().__init__(filename, vocab_map_file, batch_size, num_steps)
        with openall(self.filename) as inf:
            self.data = inf.read().replace('\n', ' </s> ').split()

    def __iter__(self):
        len_needed = self.batch_size * self.num_steps
        for i in range(0, len(self.data), len_needed):
            yield(self._convert(self.data[i:i + len_needed]))


def file_reader(filename, vocab_map_file, batch_size, num_steps,
                in_memory=False):
    """
    Returns a file reader. Parameters:
    - filename: the name of the text file to read
    - vocab_map_file: the vocabulary list file
    - batch_size: how many batches are processed at the same time
    - num_steps: the length of a single batch; the number of unrolled steps
    """
    cls = InMemoryFileReader if in_memory else StreamFileReader
    return cls(filename, vocab_map_file, batch_size, num_steps)
