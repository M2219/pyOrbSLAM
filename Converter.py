import numpy as np
import g2o

class Converter:
    def __init__(self):
        pass

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

    def to_mat(self, SE3):
        """
        Convert a g2o.SE3Quat object to a 4x4 transformation matrix (numpy array).

        Args:
            SE3 (g2o.SE3Quat): SE3 object containing rotation and translation.

        Returns:
            np.ndarray: 4x4 transformation matrix (homogeneous matrix).
        """
        # Convert SE3Quat to a 4x4 homogeneous transformation matrix
        eig_mat = SE3.to_homogeneous_matrix()

        # Ensure the result is returned as a NumPy array
        return np.array(eig_mat, dtype=np.float64)


    def sim3_to_mat(self, sim3):
        """
        Convert a 4x4 transformation matrix (cvT) to g2o.SE3Quat.

        Args:
            cvT (np.ndarray): 4x4 transformation matrix (numpy array).

        Returns:
            g2o.SE3Quat: SE3 object containing rotation and translation.
        """
        cvT = sim3.rotation().matrix()
        Tr = sim3.translation()
        s = sim3.scale()
        # Extract rotation matrix (3x3)
        Tf = np.array([
            [s * cvT[0, 0], s * cvT[0, 1], s * cvT[0, 2], Tr[0]],
            [s * cvT[1, 0], s * cvT[1, 1], s * cvT[1, 2], Tr[1]],
            [s * cvT[2, 0], s * cvT[2, 1], s * cvT[2, 2], Tr[2]],
            [0., 0., 0., 1.],
        ], dtype=np.float64)

        return Tf

    def RT_to_TF(self, cvT, Tr):
        """
        Convert a 4x4 transformation matrix (cvT) to g2o.SE3Quat.

        Args:
            cvT (np.ndarray): 4x4 transformation matrix (numpy array).

        Returns:
            g2o.SE3Quat: SE3 object containing rotation and translation.
        """
        # Extract rotation matrix (3x3)
        Tf = np.array([
            [cvT[0, 0], cvT[0, 1], cvT[0, 2], Tr[0]],
            [cvT[1, 0], cvT[1, 1], cvT[1, 2], Tr[1]],
            [cvT[2, 0], cvT[2, 1], cvT[2, 2], Tr[2]],
            [0., 0., 0., 1.],
        ], dtype=np.float64)

        return Tf

