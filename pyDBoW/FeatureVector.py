from collections import OrderedDict

class FeatureVector:
    def __init__(self):
        """Initialize an empty feature vector."""
        self.data = {}

    def add_feature(self, node_id, feature_index):
        """
        Adds a feature index to the vector of a node, or creates a new node with the feature.
        :param node_id: The ID of the node.
        :param feature_index: The index of the feature to add.
        """
        if node_id not in self.data:
            self.data[node_id] = []

        self.data[node_id].append(feature_index)
        self.data = OrderedDict(sorted(self.data.items()))
        return self.data

