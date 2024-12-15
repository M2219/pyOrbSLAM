import numpy as np

class FrameDrawer:
    def __init__(self, pMap):
        self.mpMap = pMap
        self.mState = "SYSTEM_NOT_READY"
        self.mIm = np.zeros((480, 640, 3), dtype=np.uint8)

