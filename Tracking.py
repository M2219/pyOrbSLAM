import numpy as np

import cv2

from ORBExtractor import ORBExtractor

class Tracking:
    def __init__(self, pSys, pVoc, pFrameDrawer, pMapDrawer, pMap, pKFDB, fSettings, sensor, ss):

        self.ss = ss


        self.mState = "NO_IMAGES_YET"
        self.mSensor = sensor
        self.mbOnlyTracking = False
        self.mbVO = False
        self.mpORBVocabulary = pVoc
        self.mpKeyFrameDB = pKFDB
        self.mpInitializer = None
        self.mpSystem = pSys
        self.mpViewer = None
        self.mpFrameDrawer = pFrameDrawer
        self.mpMapDrawer = pMapDrawer
        self.mpMap = pMap
        self.mnLastRelocFrameId = 0

        fx = fSettings["Camera.fx"]
        fy = fSettings["Camera.fy"]
        cx = fSettings["Camera.cx"]
        cy = fSettings["Camera.cy"]

        self.mK = np.eye(3, dtype=np.float32)
        self.mK[0, 0] = fx
        self.mK[1, 1] = fy
        self.mK[0, 2] = cx
        self.mK[1, 2] = cy

        self.mDistCoef = np.zeros((4, 1), dtype=np.float32)
        self.mDistCoef[0, 0] = fSettings["Camera.k1"]
        self.mDistCoef[1, 0] = fSettings["Camera.k2"]
        self.mDistCoef[2, 0] = fSettings["Camera.p1"]
        self.mDistCoef[3, 0] = fSettings["Camera.p2"]

        self.mbf = fSettings["Camera.bf"]
        fps = fSettings["Camera.fps"]
        self.mMaxFrames = fps if fps > 0 else 30
        self.mMinFrames = 0

        self.mbRGB = fSettings["Camera.RGB"]

        nFeatures = int(fSettings["ORBextractor.nFeatures"])
        fScaleFactor = fSettings["ORBextractor.scaleFactor"]
        nLevels = int(fSettings["ORBextractor.nLevels"])
        fIniThFAST = int(fSettings["ORBextractor.iniThFAST"])
        fMinThFAST = int(fSettings["ORBextractor.minThFAST"])

        self.mpORBextractorLeft = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        self.mpORBextractorRight = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        self.mThDepth = self.mbf * fSettings["ThDepth"] / fx




