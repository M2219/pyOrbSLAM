class LoopClosing:
    def __init__(self, pMap, pDB, pVoc, bFixScale):
        """
        Initializes the LoopClosing class.

        Args:
            pMap (Map): Pointer to the map.
            pDB (KeyFrameDatabase): Pointer to the KeyFrame database.
            pVoc (ORBVocabulary): Pointer to the ORB vocabulary.
            bFixScale (bool): Whether to fix the scale.
        """
        self.mbResetRequested = False
        self.mbFinishRequested = False
        self.mbFinished = True
        self.mpMap = pMap
        self.mpKeyFrameDB = pDB
        self.mpORBVocabulary = pVoc
        self.mpMatchedKF = None
        self.mLastLoopKFid = 0
        self.mbRunningGBA = False
        self.mbFinishedGBA = True
        self.mbStopGBA = False
        self.mpThreadGBA = None
        self.mbFixScale = bFixScale
        self.mnFullBAIdx = 0
        self.mnCovisibilityConsistencyTh = 3
