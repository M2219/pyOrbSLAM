class LocalMapping:
    def __init__(self, pMap, bMonocular):

        self.mbMonocular = bMonocular
        self.mbResetRequested = False
        self.mbFinishRequested = False
        self.mbFinished = True
        self.mpMap = pMap

        self.mbAbortBA = False
        self.mbStopped = False
        self.mbStopRequested = False
        self.mbNotStop = False
        self.mbAcceptKeyFrames = True

