#! /usr/bin/python3

import numpy as np


def vectorize_seqs(seqs, dim):
    """
    vectorize seqs to the seqs of shape (len(seqs), dim).
    """
    res = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        res[i, seq] = 1
    return res


def to_one_hot(labels, dim):
    """
    vectorize labels by one-hot encoding.
    """
    res = np.zeros((len(labels), dim))
    for i, label in enumerate(labels):
        res[i, label] = 1
    return res
