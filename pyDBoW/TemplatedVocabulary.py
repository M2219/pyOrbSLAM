import numpy as np
from collections import defaultdict

from .BowVector import BowVector
from .FeatureVector import FeatureVector
from .FORB import FORB
from .ScoringObject import GeneralScoring as gs

class Node:
    """Represents a node in the vocabulary tree."""
    def __init__(self, node_id, parent_id=None):
        self.id = node_id
        self.parent = parent_id
        self.children = []
        self.descriptor = None
        self.weight = 0.0
        self.word_id = 0

    def is_leaf(self):
        return len(self.children) == 0

class TemplatedVocabulary:
    def __init__(self, k=10, L=5, weighting="TF_IDF", scoring="L1_NORM"):
        """
        Initializes the vocabulary tree with descriptor length of 32
        :param k: Branching factor.
        :param L: Depth levels.
        :param weighting: Weighting scheme ("TF", "TF_IDF", "IDF", "BINARY").
        :param scoring: Scoring method ("L1_NORM", "L2_NORM", etc.).
        """
        self.k = k
        self.L = L
        self.weighting = weighting
        self.scoring = scoring
        self.nodes = [Node(0)]
        self.words = []
        self.node_count = 1

        self.scoring_object = None
        self.create_scoring_object()
        self.forb = FORB(32)

    def load_from_text_file(self, filename):
        with open(filename, "r") as f:
            header = f.readline().strip().split()
            self.k = int(header[0])
            self.L = int(header[1])
            n1 = int(header[2])
            n2 = int(header[3])

            if self.k < 0 or self.k > 20 or self.L < 1 or self.L > 10 or n1 < 0 or n1 > 5 or n2 < 0 or n2 > 3:
                print("Vocabulary loading failure: Invalid parameters in file!")
                return False

            self.scoring = n1
            self.weighting = n2
            self.nodes = [Node(0, 0)]
            self.words = []

            for line in f:
                parts = line.strip().split()
                parent_id = int(parts[0])
                is_leaf = int(parts[1])
                descriptor_values = list(map(float, parts[2:-1]))
                weight = float(parts[-1])

                node_id = len(self.nodes)
                node = Node(node_id, parent_id)
                node.descriptor = np.array(descriptor_values).astype(int)
                node.weight = weight

                self.nodes.append(node)
                self.nodes[parent_id].children.append(node_id)

                if is_leaf > 0:
                    word_id = len(self.words)
                    node.word_id = word_id
                    self.words.append(node)
                else:
                    node.word_id = 0
        return True

    def create_scoring_object(self):
        """Creates the scoring object based on the scoring type."""
        scoring_map = {
            "L1_NORM": gs.l1_score,
            "L2_NORM": gs.l2_score,
            "CHI_SQUARE": gs.chi_square_score,
            "KL": gs.kl_score,
            "BHATTACHARYYA": gs.bhattacharyya_score,
            "DOT_PRODUCT": gs.dot_product_score,
        }
        if self.scoring in scoring_map:
            self.scoring_object = scoring_map[self.scoring]

    def size(self):
        return len(self.words)

    def score(self, bow_vector1, bow_vector2):
        """
        Computes the score between two BoW vectors using the scoring object.
        :param bow_vector1: First BoW vector (dictionary with word IDs as keys and weights as values).
        :param bow_vector2: Second BoW vector (dictionary with word IDs as keys and weights as values).
        :return: Score (similarity or distance) between the two vectors.
        """
        return self.scoring_object(bow_vector1, bow_vector2)

    def transform(self, features, levels_up=4):
        """
        Transforms descriptors into a BoW vector and FeatureVector.
        :param features: List of descriptors.
        :param levels_up: Levels to go up the vocabulary tree for node IDs.
        :return: (BoW vector, FeatureVector)
        """
        bv = BowVector()
        fv = FeatureVector()
        mustNormalize = True
        node_id = 0

        for i in range(features.shape[0]):
            word_id, node_id, weight = self.transform_feature(features[i], node_id, levels_up)
            if weight > 0:
                bv.add_weight(word_id, weight)
                fv.add_feature(node_id, i)

        if mustNormalize:
            bv.normalize(norm_type="L1")

        return bv.word_weights, fv.data

    def transform_feature(self, feature, nid,  levels_up):
        """
        Finds the word ID for a given feature by traversing the tree.
        :param feature: A single descriptor.
        :param levels_up: Levels to go up the vocabulary tree.
        :return: (word_id, node_id, weight)
        """
        level = 0
        final_id = 0
        nid_level = self.L - levels_up

        while not self.nodes[final_id].is_leaf():
            node_s = self.nodes[final_id].children
            final_id = node_s[0]
            best_d = self.forb.distance(feature, self.nodes[final_id].descriptor)
            for nit in node_s[1:]:
                d = self.forb.distance(feature, self.nodes[nit].descriptor)

                if d < best_d:
                    best_d = d
                    final_id = nit

            level += 1

            if (nid is not None) and (level == nid_level):
                nid = final_id

        word_id = self.nodes[final_id].word_id
        weight = self.nodes[final_id].weight

        return word_id, nid, weight


