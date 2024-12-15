class LoopClosing:
    def __init__(self, pMap, pDB, pVoc, bFixScale, ss):


        self.ss = ss
        #mutex



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
