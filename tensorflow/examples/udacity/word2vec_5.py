#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Exercises for assignment 5 -- word2vec."""
from __future__ import print_function
from argparse import ArgumentParser
import collections
from functools import partial
import gzip
import math
import random
import time

import numpy as np
import tensorflow as tf

from text_data import read_as_list


def parse_arguments():
    parser = ArgumentParser(
        description='Exercises for assignment 5 -- word2vec.')
    parser.add_argument('exercise', choices=['sg', 'cb'],
                        help='the exercise to run (sg, cb).')
    parser.add_argument('data_file', help='the pickled data file.')
    parser.add_argument('--vocab-size', '-v', type=int, default=50000,
                        help='the vocabulary size [50k].')
    parser.add_argument('--neg-samples', '-n', type=int, default=64,
                        help='number of negative examples to sample [64].')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='the training batch size [128].')
    parser.add_argument('--embed-size', '-e', type=int, default=128,
                        help='dimension of the embedding vector [128].')
    parser.add_argument('--window-size', type=int, default=1,
                        help='how many words to consider left and right [1].')
    parser.add_argument('--num-skips', type=int, default=2,
                        help='how many times to reuse an input to generate '
                             'a label [2].')
    parser.add_argument('--valid-size', type=int, default=16,
                        help='random set of words to evaluate similarity on [16].')
    parser.add_argument('--valid-window', type=int, default=100,
                        help='only pick dev samples in the head of the '
                             'distribution [100].')
    parser.add_argument('--iterations', type=int, default=100001,
                        help='the default number of iterations [130001].')
    parser.add_argument('--display-results', '-d', action='store_true',
                        help='whether to display the result in a plot')
    parser.add_argument('--output', '-o',
                        help='the output file (prefix, [sg,cg].txt will be '
                             'appended to it.')
    args = parser.parse_args()

    return (args.exercise, args.data_file, args.vocab_size, args.neg_samples,
            args.batch_size, args.embed_size, args.window_size, args.num_skips,
            args.valid_size, args.valid_window, args.iterations, args.output)


def build_dataset(words, vocab_size):
    """Builds the training dataset: converts words to ids."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = {v: k for k, v in dictionary.items()}
    # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def get_span(window_size):
    """[ window_size target window_size ]"""
    return window_size * 2 + 1


class DataEnumerator(collections.deque):
    def __init__(self, data, window_size):
        span = get_span(window_size)
        super(DataEnumerator, self).__init__(data[:span], maxlen=span)
        self.data = data
        self.i = span

    def next(self):
        self.append(self.data[self.i])
        self.i = (self.i + 1) % len(self.data)


def generate_cb_batch(data_it, batch_size, window_size):
    """
    Generates a training batch.
      - data_it: a looping iterator over the data
      - batch size: obvious
      - window_size: the window to the left and right of the target word
    """
    batch = np.ndarray(shape=(2 * window_size * batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    for i in range(batch_size):
        ith_batch = i * 2 * window_size
        for j in range(window_size):
            batch[ith_batch] = data_it[j]
            batch[ith_batch + window_size] = data_it[j + window_size + 1]
            ith_batch += 1
        labels[i] = data_it[window_size]
        data_it.next()
    return batch, labels


def generate_sg_batch(data_it, batch_size, window_size, num_skips):
    """
    Generates a training batch.
      - data_it: a looping iterator over the data
      - batch size: obvious
      - window_size: the window to the left and right of the target word
      - num_skips: how many skip-grams per word (consequently,
        batch_size // num_skips words in a batch)
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * window_size
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = get_span(window_size)
    # Add num_skips source -> target skip-grams per target word, then
    # go to the next word
    for i in range(batch_size // num_skips):
        target = window_size  # target label at the center of the buffer
        targets_to_avoid = [window_size]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = data_it[window_size]
            labels[i * num_skips + j, 0] = data_it[target]
        data_it.next()
    return batch, labels


def example_cb_batch(data, reverse_dictionary):
    """Generates and prints a batch -- much easier to understand it this way."""
    print('data:', [reverse_dictionary[di] for di in data[:10]])
    window_size = 3
    batch, labels = generate_cb_batch(
        DataEnumerator(data, window_size), batch_size=4, window_size=window_size)
    print('\nwith window_size = %d:' % (window_size))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(4)])


def example_sg_batch(data, reverse_dictionary):
    """Generates and prints a batch -- much easier to understand it this way."""
    print('data:', [reverse_dictionary[di] for di in data[:20]])
    for num_skips, window_size in [(2, 1), (4, 2)]:
        batch, labels = generate_sg_batch(
            DataEnumerator(data, window_size), batch_size=8,
            num_skips=num_skips, window_size=window_size)
        print('\nwith num_skips = %d and window_size = %d:' % (num_skips, window_size))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


def create_graph(params, data_it, reverse_dictionary, sg=True):
    graph = tf.Graph()
    with graph.as_default():  # , tf.device('/cpu:0'):
        # Input data.
        if sg:
            print('SG')
            train_dataset = tf.placeholder(tf.int32, shape=[params.batch_size])
            generate_batch = partial(generate_sg_batch,
                                     num_skips=params.num_skips)
        else:
            print('CB')
            train_dataset = tf.placeholder(
                tf.int32, shape=[2 * params.window_size * params.batch_size])
            generate_batch = generate_cb_batch
        train_labels = tf.placeholder(tf.int32, shape=[params.batch_size, 1])
        valid_dataset = tf.constant(params.valid_examples, dtype=tf.int32)

        # Variables.
        embeddings = tf.Variable(
            tf.random_uniform([params.vocab_size, params.embed_size], -1.0, 1.0))
        softmax_weights = tf.Variable(
            tf.truncated_normal([params.vocab_size, params.embed_size],
                                stddev=1.0 / math.sqrt(params.embed_size)))
        softmax_biases = tf.Variable(tf.zeros([params.vocab_size]))

        # Model.
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        if not sg:
            # Average the 2 * window_size training samples for CBOW
            embed = tf.reduce_mean(
                tf.reshape(embed, [params.batch_size, 2 * params.window_size,
                                   params.embed_size]),
                1)
        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                       train_labels, params.neg_samples,
                                       params.vocab_size))

        # Optimizer.
        # Note: The optimizer will optimize the softmax_weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, tf.transpose(normalized_embeddings))

    # Training
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        average_loss = 0
        t = time.time()
        for step in range(params.iterations):
            batch_data, batch_labels = generate_batch(
                data_it, params.batch_size, params.window_size)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    time_str = ' (in {} seconds)'.format(time.time() - t)
                    t = time.time()
                else:
                    time_str = ''
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f%s' % (step, average_loss, time_str))
                average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(params.valid_size):
                    valid_word = reverse_dictionary[params.valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings


def main():
    (exercise, data_file, vocab_size, neg_samples,
     batch_size, embed_size, window_size, num_skips,
     valid_size, valid_window, iterations, output) = parse_arguments()

    # For valid evaluation
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    params = collections.namedtuple(
        'Params', ['vocab_size', 'embed_size', 'window_size', 'num_skips',
                   'neg_samples', 'batch_size', 'valid_size', 'valid_examples',
                   'iterations']
    )(vocab_size, embed_size, window_size, num_skips, neg_samples,
      batch_size, valid_size, valid_examples, iterations)

    # Read the text
    words = read_as_list(data_file)
    print('Data size %d' % len(words))

    # Convert it to a sequence of numbers
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocab_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.

    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.

    data_it = DataEnumerator(data, 2 * window_size + 1)
    if exercise == 'sg':
        # example_sg_batch(data, reverse_dictionary)
        final_embeddings = create_graph(params, data_it, reverse_dictionary)
    else:
        # example_cb_batch(data, reverse_dictionary)
        final_embeddings = create_graph(
            params, data_it, reverse_dictionary, sg=False)

    if output:
        with gzip.open('{}.{}.txt.gz'.format(output, exercise), 'wt', 5) as outf:
            for i, emb in enumerate(final_embeddings):
                print('{} {}'.format(reverse_dictionary[i],
                                     ' '.join(str(e) for e in emb)), file=outf)


if __name__ == '__main__':
    main()
