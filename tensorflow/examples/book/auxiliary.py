#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Auxiliary classes and methods."""
import bz2
import gzip

class AttrDict(dict):
    """Makes our life easier."""
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError('key {} missing'.format(key))
        self[key] = value


# TODO Put this somewhere else
def openall(
    filename, mode='rt', encoding=None, errors=None, newline=None,
    buffering=-1, closefd=True,  # , opener=None,  # for open()
    compresslevel=5,  # faster default compression
):
    """
    Opens all file types known to the Python SL. There are some differences
    from the stock functions:
    - the default mode is 'rt'
    - the default compresslevel is 5, because e.g.gzip does not benefit a lot
      from higher values, only becomes slower.
    """
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, compresslevel,
                         encoding, errors, newline)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode, compresslevel,
                        encoding, errors, newline)
    else:
        return open(filename, mode, buffering, encoding, errors, newline,
                    closefd)  # , opener)
