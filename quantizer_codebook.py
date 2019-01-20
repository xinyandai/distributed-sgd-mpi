import numpy as np
import tensorflow as tf

from vecs_io import fvecs_read
from myutils import normalize

from scipy.cluster.vq import vq
import logging


class CodebookCompressor(object):
    def __init__(self, size, shape, c_dim=16, ks=256):
        self.Ks = ks
        self.size = size
        self.shape = shape
        self.dim = c_dim if self.size >= c_dim else self.size
        self.code_dtype = np.uint8 if self.Ks <= 2 ** 8 else (np.uint16 if self.Ks <= 2 ** 16 else np.uint32)

        self.M = size // self.dim
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(self.dim, self.dim)
        _, self.codewords = normalize(fvecs_read('./codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.Ks)))

    def compress(self, vec):
        vec = vec.reshape((-1, self.dim))
        norms, normalized_vecs = normalize(vec)
        codes, _ = vq(normalized_vecs, self.codewords)
        return [norms, codes.astype(self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature
        vec = np.empty((len(norms), self.dim), dtype=np.float32)
        vec[:, :] = self.codewords[codes[:], :]
        vec[:, :] = (vec.transpose() * norms).transpose()

        return vec.reshape(self.shape)


