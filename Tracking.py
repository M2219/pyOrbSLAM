import cv2
import numpy as np

class Tracking:
    def __init__(self, pSys, pVoc, pFrameDrawer, pMapDrawer, pMap, pKFDB, strSettingPath, sensor):
        """
        Initializes the Tracking class.

        Args:
            pSys (System): Pointer to the system.
            pVoc (ORBVocabulary): Pointer to the ORB vocabulary.
            pFrameDrawer (FrameDrawer): Pointer to the frame drawer.
            pMapDrawer (MapDrawer): Pointer to the map drawer.
            pMap (Map): Pointer to the map.
            pKFDB (KeyFrameDatabase): Pointer to the KeyFrame database.
            strSettingPath (str): Path to the settings file.
            sensor (int): Sensor type (e.g., MONOCULAR, STEREO, RGBD).
        """
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

        # Load camera parameters
        fSettings = cv2.FileStorage(strSettingPath, cv2.FILE_STORAGE_READ)
        fx = fSettings.getNode("Camera.fx").real()
        fy = fSettings.getNode("Camera.fy").real()
        cx = fSettings.getNode("Camera.cx").real()
        cy = fSettings.getNode("Camera.cy").real()

        self.mK = np.eye(3, dtype=np.float32)
        self.mK[0, 0] = fx
        self.mK[1, 1] = fy
        self.mK[0, 2] = cx
        self.mK[1, 2] = cy

        self.mDistCoef = np.zeros((4, 1), dtype=np.float32)
        self.mDistCoef[0, 0] = fSettings.getNode("Camera.k1").real()
        self.mDistCoef[1, 0] = fSettings.getNode("Camera.k2").real()
        self.mDistCoef[2, 0] = fSettings.getNode("Camera.p1").real()
        self.mDistCoef[3, 0] = fSettings.getNode("Camera.p2").real()
        k3 = fSettings.getNode("Camera.k3").real()
        if k3 != 0:
            self.mDistCoef = np.vstack((self.mDistCoef, np.array([[k3]], dtype=np.float32)))

        self.mbf = fSettings.getNode("Camera.bf").real()
        fps = fSettings.getNode("Camera.fps").real()
        self.mMaxFrames = fps if fps > 0 else 30
        self.mMinFrames = 0

        self.mbRGB = fSettings.getNode("Camera.RGB").real()

        # Load ORB parameters
        nFeatures = int(fSettings.getNode("ORBextractor.nFeatures").real())
        fScaleFactor = fSettings.getNode("ORBextractor.scaleFactor").real()
        nLevels = int(fSettings.getNode("ORBextractor.nLevels").real())
        fIniThFAST = int(fSettings.getNode("ORBextractor.iniThFAST").real())
        fMinThFAST = int(fSettings.getNode("ORBextractor.minThFAST").real())

        self.mpORBextractorLeft = ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        if sensor in ["STEREO", "RGBD"]:
            self.mpORBextractorRight = ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        if sensor == "MONOCULAR":
            self.mpIniORBextractor = ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)

        if sensor in ["STEREO", "RGBD"]:
            self.mThDepth = self.mbf * fSettings.getNode("ThDepth").real() / fx

        if sensor == "RGBD":
            self.mDepthMapFactor = fSettings.getNode("DepthMapFactor").real()
            self.mDepthMapFactor = 1.0 / self.mDepthMapFactor if abs(self.mDepthMapFactor) > 1e-5 else 1.0

        fSettings.release()
