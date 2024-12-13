
import numpy as np

class Initializer:
    def __init__(self, ReferenceFrame, sigma, iterations):
        """
        Initializes the Initializer.

        Args:
            ReferenceFrame (Frame): The reference frame.
            sigma (float): Sigma value for initialization.
            iterations (int): Maximum number of iterations.
        """
        self.mK = ReferenceFrame.mK.copy()  # Clone the camera matrix
        self.mvKeys1 = ReferenceFrame.mvKeysUn  # Unprojected keypoints from the reference frame

        self.mSigma = sigma
        self.mSigma2 = sigma * sigma
        self.mMaxIterations = iterations
