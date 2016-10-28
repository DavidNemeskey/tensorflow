#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Cuts the input (files) into batch_size consecutive chunks; these include
the EoS </s> token.
"""

class DataLoader(object):
    def __init__(self, header, batch_size, num_steps, one_hot=False,
                 data_type=np.int32, vocab_file=None):
        self.header = header
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.one_hot = one_hot
        self.data_type = data_type
        self.vocab = self._read_vocab(vocab_file) if vocab_file else None

    def __iter__(self):
        raise NotImlementedError('__iter__ must be implemented.')


def data_loader(header):
    with openall(header) as inf:
        fields = inf.readline().strip().split('\t')
        if fields[0] == 'TXT_DISK':
            cls = TxtDiskLoader
            data_fields[1:]
        else:
            cls = IntMemLoader

    return int(data_len)

