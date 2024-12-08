import numpy as np
class GeneralScoring:
    def __init__(self):
        """Class implementing multiple scoring methods for BowVectors."""

    @staticmethod
    def l1_score(v1, v2):
        """
        Computes the L1 norm similarity score between two vectors.
        :param v1: First BowVector as a dictionary.
        :param v2: Second BowVector as a dictionary.
        :return: L1 similarity score.
        """
        keys_v1 = set(v1.keys())
        keys_v2 = set(v2.keys())
        shared_keys = keys_v1 & keys_v2

        score = 0.0

        for key in shared_keys:
            vi = v1[key]
            wi = v2[key]
            score += abs(vi - wi) - abs(vi) - abs(wi)

        score = -score / 2.0

        return score

    @staticmethod
    def l2_score(v1, v2):
        """Computes the L2 norm similarity score."""
        shared_keys = set(v1.keys()) & set(v2.keys())
        dot_product = sum(v1[k] * v2[k] for k in shared_keys)
        if dot_product >= 1.0:
            return 1.0
        return 1.0 - np.sqrt(1.0 - dot_product)

    @staticmethod
    def chi_square_score(v1, v2):
        """
        Computes the Chi-Square similarity score between two vectors.
        :param v1: First BowVector as a dictionary.
        :param v2: Second BowVector as a dictionary.
        :return: Chi-Square similarity score.
        """
        shared_keys = set(v1.keys()) & set(v2.keys())
        score = 0.0

        for key in shared_keys:
            vi = v1[key]
            wi = v2[key]
            if vi + wi != 0.0:
                score += (vi * wi) / (vi + wi)

        score *= 2.0

        return score

    @staticmethod
    def kl_score(v1, v2):
        """
        Computes the Kullback-Leibler divergence between two vectors.
        :param v1: First BowVector as a dictionary.
        :param v2: Second BowVector as a dictionary.
        :return: KL divergence.
        """
        LOG_EPS = np.log(np.finfo(float).eps)
        score = 0.0

        for k in v1.keys():
            vi = v1[k]
            wi = v2.get(k, 0)
            if wi > 0 and vi > 0:
                score += vi * np.log(vi / wi)
            elif vi > 0:
                score += vi * (np.log(vi) - LOG_EPS)

        return score

    @staticmethod
    def bhattacharyya_score(v1, v2):
        """Computes the Bhattacharyya similarity score."""
        shared_keys = set(v1.keys()) & set(v2.keys())
        score = sum(np.sqrt(v1[k] * v2[k]) for k in shared_keys)
        return score

    @staticmethod
    def dot_product_score(v1, v2):
        """Computes the dot product similarity score."""
        shared_keys = set(v1.keys()) & set(v2.keys())
        return sum(v1[k] * v2[k] for k in shared_keys)


if __name__ == "__main__":

  import subprocess
  from collections import defaultdict
  from BowVector import BowVector

  g = GeneralScoring()
  # Python Scoring Classes
  # Sample BowVectors
  v1 = BowVector()
  v2 = BowVector()

  v1.add_weight(1, 0.1);
  v1.add_weight(1, 0.03);
  v1.add_weight(2, 0.2);
  v1.add_weight(3, 0.37);
  v1.add_weight(4, 0.343);
  v1.add_weight(5, 0.32);
  v1.add_weight(6, 0.37);

  print(v1.word_weights)

  v2.add_weight(1, 0.12);
  v2.add_weight(1, 0.18);
  v2.add_weight(3, 0.423);
  v2.add_weight(4, 0.516);
  v2.add_weight(7, 0.526);
  v2.add_weight(8, 0.566);
  v2.add_weight(9, 0.576);

  print(v2.word_weights)

  scorers = {
      "L1": g.l1_score(v1.word_weights, v2.word_weights),
      "L2": g.l2_score(v1.word_weights, v2.word_weights),
      "ChiSquare": g.chi_square_score(v1.word_weights, v2.word_weights),
      "KL": g.kl_score(v1.word_weights, v2.word_weights),
      "Bhattacharyya": g.bhattacharyya_score(v1.word_weights, v2.word_weights),
      "DotProduct": g.dot_product_score(v1.word_weights, v2.word_weights),
  }


  print("Python Scoring Results:")
  for name, score in scorers.items():
      print(f"{name} Score: {score:.4f}")

