import numpy as np

class FrameDrawer:
    def __init__(self, pMap):
        """
        Initializes the FrameDrawer.

        Args:
            pMap (Map): The map object.
        """
        self.mpMap = pMap
        self.mState = "SYSTEM_NOT_READY"  # Initial tracking state
        self.mIm = np.zeros((480, 640, 3), dtype=np.uint8)  # Blank image (480x640, 3 channels, black)

