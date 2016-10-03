#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Fetches abstracts from ArXiv."""

from argparse import ArgumentParser
import os
import random

import bs4
import numpy as np
import requests


class ArXivAbstracts(object):
    ENDPOINT = 'http://export.arxiv.org/api/query'
    PAGE_SIZE = 100

    def __init__(self, cache_dir, categories, keywords, amount=None):
        """Checks if we have already downloaded something, else fetches."""
        self.categories = categories
        self.keywords = keywords
        cache_dir = os.path.expanduser(cache_dir)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        filename = os.path.join(cache_dir, 'abstracts.txt')
        if not os.path.isfile(filename):
            try:
                with open(filename, 'w') as file_:
                    for abstract in self._fetch_all(amount):
                        file_.write(abstract + '\n')
            except:
                if os.path.isfile(filename):
                    os.remove(filename)
                raise
        with open(filename) as file_:
            self.data = [l.strip() for l in file_.readlines()]

    def _fetch_all(self, amount):
        """Implements pagination over SERPs."""
        page_size = type(self).PAGE_SIZE
        count = self._fetch_count()
        if amount:
            count = min(count, amount)
        for offset in range(0, count, page_size):
            print('Fetch papers {}/{}'.format(offset + page_size, count))
            yield from self._fetch_page(page_size, count)

    def _fetch_page(self, amount, offset):
        url = self._build_url(amount, offset)
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text)
        for entry in soup.findAll('entry'):
            text = entry.find('summary').text
            text = text.strip().replace('\n', ' ')
            yield text

    def _fetch_count(self):
        url = self._build_url(0, 0)
        response = requests.get(url)
        soup = bs4.BeautifulSoup(response.text, 'lxml')
        count = int(soup.find('opensearch:totalresults').string)
        print(count, 'papers found')
        return count

    def _build_url(self, amount, offset):
        categories = ' OR '.join('cat:' + x for x in self.categories)
        keywords = ' OR '.join('all:' + x for x in self.keywords)
        url = type(self).ENDPOINT
        url += '?search_query=(({}) AND ({}))'.format(categories, keywords)
        url += '&max_results={}&offset={}'.format(amount, offset)
        return url


class Preprocessing:
    VOCABULARY = " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
                 "\\^_abcdefghijklmnopqrstuvwxyz{|}"

    def __init__(self, texts, length, batch_size):
        self.texts = texts
        self.length = length
        self.batch_size = batch_size
        self.lookup = {x: i for i, x in enumerate(self.VOCABULARY)}

    def __call__(self, texts):
        batch = np.zeros((len(texts), self.length, len(self.VOCABULARY)))
        for index, text in enumerate(texts):
            text = [x for x in text if x in self.lookup]
            assert 2 <= len(text) <= self.length
            for offset, character in enumerate(text):
                code = self.lookup[character]
                batch[index, offset, code] = 1
        return batch

    def __iter__(self):
        windows = []
        for text in self.texts:
            for i in range(0, len(text) - self.length + 1, self.length // 2):
                windows.append(text[i: i + self.length])
                print(windows[-1])
        assert all(len(x) == len(windows[0]) for x in windows)
        while True:
            random.shuffle(windows)
            for i in range(0, len(windows), self.batch_size):
                batch = windows[i: i + self.batch_size]
                yield self(batch)


def parse_arguments():
    parser = ArgumentParser(
        description='Fetches abstracts from ArXiv.')
    parser.add_argument('--category', '-c', action='append', default=[],
                        help='the categories (can specify more than once). ' +
                             'Note that this must be the ArXiv designation, '
                             'not meaningful words.')
    parser.add_argument('--query', '-q', action='append', default=[],
                        help='the query words (can specify more than once).')
    parser.add_argument('--cache-dir', '-d', default='.',
                        help='the cache directory [.].')
    args = parser.parse_args()

    if not len(args.query):
        parser.error('At least one query word must be specified.')
    return args.category, args.query, args.cache_dir


def main():
    category, query, cache_dir = parse_arguments()
    ArXivAbstracts(cache_dir, category, query)


if __name__ == '__main__':
    main()
