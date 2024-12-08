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

if __name__ == "__main__":

    # Python equivalent test
    python_fv = FeatureVector()
    a=python_fv.addFeature(1, 10)
    print(a)

    a1=python_fv.addFeature(1, 20)
    print(a1)

    a2=python_fv.addFeature(2, 30)
    print(a2)

    a3=python_fv.addFeature(3, 40)
    print(a3)

    a4=python_fv.addFeature(2, 50)
    print(a4)

