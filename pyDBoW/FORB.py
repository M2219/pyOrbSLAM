import numpy as np

class FORB:
    def __init__(self, L):
        self.L = 32

    def meanValue(self, descriptors):
        if len(descriptors) == 0:
            return None
        if len(descriptors) == 1:
            return descriptors[0].copy()

        sum_bits = np.zeros(self.L * 8, dtype=np.int32)
        for desc in descriptors:
            for byte_idx, byte in enumerate(desc):
                for bit_idx in range(8):
                    if byte & (1 << (7 - bit_idx)):
                        sum_bits[byte_idx * 8 + bit_idx] += 1

        N2 = len(descriptors) // 2 + len(descriptors) % 2
        mean_descriptor = np.zeros(self.L, dtype=np.uint8)
        for i, count in enumerate(sum_bits):
            if count >= N2:
                byte_idx = i // 8
                bit_idx = i % 8
                mean_descriptor[byte_idx] |= (1 << (7 - bit_idx))

        return mean_descriptor

    def distance(self, a, b):
        xor = np.bitwise_xor(a, b)
        return sum(bin(byte).count('1') for byte in xor)

    def toString(self, descriptor):
        return " ".join(map(str, descriptor.tolist()))

    def fromString(self, s):
        return np.array(list(map(int, s.split())), dtype=np.uint8)

    def toMat32F(self, descriptors):
        mat = np.zeros((len(descriptors), self.L * 8), dtype=np.float32)
        for i, desc in enumerate(descriptors):
            for j, byte in enumerate(desc):
                for k in range(8):
                    mat[i, j * 8 + k] = 1 if byte & (1 << (7 - k)) else 0
        return mat

    def toMat8U(self, descriptors):
        return np.array(descriptors, dtype=np.uint8)



