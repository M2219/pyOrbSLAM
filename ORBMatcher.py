import numpy as np
class ORBMatcher:
    def __init__(self):

        self.TH_HIGH = 100
        self.TH_LOW = 50
        self.HISTO_LENGTH = 30

    def descriptor_distance(self, a, b):
        xor = np.bitwise_xor(a, b)
        return sum(bin(byte).count('1') for byte in xor)





if __name__ == "__main__":

    ORb = ORBMatcher()
    a = np.array([3, 191, 24, 185, 182, 169, 189, 31, 30, 9, 90, 231, 181, 10, 192, 0, 183, 27, 31, 149, 147, 152, 69, 127, 172, 4, 62, 192, 156, 72, 129, 153])
    b = np.array([90, 215, 107, 14, 210, 82, 63, 189, 49, 76, 223, 231, 73, 102, 158, 213, 54, 215, 237, 230, 253, 154, 173, 125, 50, 99, 170, 196, 47, 166, 175, 85])
    print(ORb.distance(a, b))
