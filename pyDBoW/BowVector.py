import numpy as np
from collections import OrderedDict

class BowVector:
    def __init__(self):
        self.word_weights = {}

    def add_weight(self, word_id, weight):
        """Add or increment the weight of a word."""
        if word_id in self.word_weights:
            self.word_weights[word_id] += weight
        else:
            self.word_weights[word_id] = weight

        self.word_weights = OrderedDict(sorted(self.word_weights.items()))

    def add_if_not_exist(self, word_id, weight):
        """Add the word only if it doesn't already exist."""
        if word_id not in self.word_weights:
            self.word_weights[word_id] = weight

    def normalize(self, norm_type="L1"):
        """Normalize the vector using L1 or L2 norm."""
        if norm_type == "L1":
            total_weight = sum(self.word_weights.values())
            if total_weight > 0:
                for word_id in self.word_weights:
                    self.word_weights[word_id] /= total_weight

        elif norm_type == "L2":
            total_weight = np.sqrt(sum(w ** 2 for w in self.word_weights.values()))
            if total_weight > 0:
                for word_id in self.word_weights:
                    self.word_weights[word_id] /= total_weight

        else:
            raise ValueError("Unsupported normalization type. Use 'L1' or 'L2'.")

    def save_m(self, filename, vocab_size):
        """Save the vector in MATLAB-compatible format."""
        with open(filename, "w") as f:
            vec = [self.word_weights.get(i, 0.0) for i in range(vocab_size)]
            f.write(" ".join(map(str, vec)) + "\n")
