import numpy as np
import g2o

class Convertor:
    def __init__(self):
       print("convertor")

    def to_descriptor_vector(self, descriptors):
        return descriptors.reshape(-1)

    def to_se3_quat(self, cvT):
        """
        Convert a 4x4 transformation matrix (cvT) to g2o.SE3Quat.

        Args:
            cvT (np.ndarray): 4x4 transformation matrix (numpy array).

        Returns:
            g2o.SE3Quat: SE3 object containing rotation and translation.
        """
        # Extract rotation matrix (3x3)
        R = np.array([
            [cvT[0, 0], cvT[0, 1], cvT[0, 2]],
            [cvT[1, 0], cvT[1, 1], cvT[1, 2]],
            [cvT[2, 0], cvT[2, 1], cvT[2, 2]],
        ], dtype=np.float64)

        # Extract translation vector (3x1)
        t = np.array([cvT[0, 3], cvT[1, 3], cvT[2, 3]], dtype=np.float64)

        # Create and return g2o.SE3Quat
        return g2o.SE3Quat(R, t)
