#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Exercises for assignment 5 -- word2vec."""
from __future__ import print_function
from argparse import ArgumentParser
import collections
import math
import random

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
    parser.add_argument('--skip-window', type=int, default=1,
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
                        help='the default number of iterations [100001].')
    parser.add_argument('--display-results', '-d', action='store_true',
                        help='whether to display the result in a plot')
    args = parser.parse_args()

    return (args.exercise, args.data_file, args.vocab_size, args.neg_samples,
            args.batch_size, args.embed_size, args.skip_window, args.num_skips,
            args.valid_size, args.valid_window, args.iterations)


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


def generate_batch(data, data_index, batch_size, num_skips, skip_window):
    """
    Generates a training batch.
      - data: obvious
      - data_index: the global index counter
      - batch size: obvious
      - num_skips: how many skip-grams per word (consequently,
        batch_size // num_skips words in a batch)
      - skip_window: the window to the left and right of the target word
    """
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        # WTF the beginning and the end are neighbors?!
        data_index = (data_index + 1) % len(data)
    # Add num_skips source -> target skip-grams per target word, then
    # go to the next word
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, data_index


def example_batch(data, reverse_dictionary):
    """Generates and prints a batch -- much easier to understand it this way."""
    print('data:', [reverse_dictionary[di] for di in data[:8]])
    for num_skips, skip_window in [(2, 1), (4, 2)]:
        data_index = 0
        batch, labels, data_index = generate_batch(
            data, data_index, batch_size=8,
            num_skips=num_skips, skip_window=skip_window)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


def create_sg_graph(graph, params, data, reverse_dictionary):
    with graph.as_default():  # , tf.device('/cpu:0'):
        # Input data.
        train_dataset = tf.placeholder(tf.int32, shape=[params.batch_size])
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
        data_index = 0
        for step in range(params.iterations):
            batch_data, batch_labels, data_index = generate_batch(
                data, data_index, params.batch_size,
                params.num_skips, params.skip_window)
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
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
     batch_size, embed_size, skip_window, num_skips,
     valid_size, valid_window, iterations) = parse_arguments()

    # For valid evaluation
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    params = collections.namedtuple(
        'Params', ['vocab_size', 'embed_size', 'skip_window', 'num_skips',
                   'neg_samples', 'batch_size', 'valid_size', 'valid_examples',
                   'iterations']
    )(vocab_size, embed_size, skip_window, num_skips, neg_samples,
      batch_size, valid_size, valid_examples, iterations)

    # Read the text
    words = read_as_list(data_file)
    print('Data size %d' % len(words))

    # Convert it to a sequence of numbers
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocab_size)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.

    example_batch(data, reverse_dictionary)

    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))

    graph = tf.Graph()
    create_sg_graph(graph, params, data, reverse_dictionary)


if __name__ == '__main__':
    main()
