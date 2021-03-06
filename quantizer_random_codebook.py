import numpy as np
import tensorflow as tf

from vecs_io import fvecs_read
from myutils import normalize


class RandomCodebookCompressor(object):
    def __init__(self, size, shape, c_dim=32, k=128):
        self.Ks = k // 2
        self.size = size
        self.shape = shape
        self.dim = c_dim if self.size >= c_dim else self.size
        self.code_dtype = np.uint8 if self.Ks <= 2 ** 8 else (np.uint16 if self.Ks <= 2 ** 16 else np.uint32)

        self.M = size // self.dim
        assert size % self.dim == 0, \
            "dimension of variable should be smaller than {} or dividable by {}".format(self.dim, self.dim)
        _, self.codewords = normalize(fvecs_read('./codebook/angular_dim_{}_Ks_{}.fvecs'.format(self.dim, self.Ks)))
        self.c = self.codewords.T
        self.c_dagger = np.linalg.pinv(self.c)
        self.codewords = np.concatenate((self.codewords, -self.codewords))

    def compress(self, vec):
        vec = vec.reshape((-1, self.dim))
        bar_p = (self.c_dagger @ vec.T).T
        l1_norms, normalized_vecs = normalize(bar_p, order=1)
        tild_p = np.clip(np.concatenate(
            (normalized_vecs, -normalized_vecs), axis=1), 0, 1)

        r = np.random.uniform(0, 1, size=(tild_p.shape[0], 1))
        codes = np.argmax(np.cumsum(tild_p, axis=1) >
                          np.tile(r, (1, tild_p.shape[1])), axis=1)
        return [l1_norms, np.array(codes).astype(self.code_dtype)]

    def decompress(self, signature):
        [norms, codes] = signature
        vec = np.empty((len(norms), self.dim), dtype=np.float32)
        vec[:, :] = self.codewords[codes[:], :]
        vec[:, :] = (vec.transpose() * norms).transpose()

        return vec.reshape(self.shape)

