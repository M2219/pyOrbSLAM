import cv2

class Viewer:
    def __init__(self, pSystem, pFrameDrawer, pMapDrawer, pTracking, strSettingPath, ss):

        self.ss = ss
        # mutex

        self.mpSystem = pSystem
        self.mpFrameDrawer = pFrameDrawer
        self.mpMapDrawer = pMapDrawer
        self.mpTracker = pTracking

        self.mbFinishRequested = False
        self.mbFinished = True
        self.mbStopped = True
        self.mbStopRequested = False

        # Load settings from file
        fSettings = cv2.FileStorage(strSettingPath, cv2.FILE_STORAGE_READ)

        fps = fSettings.getNode("Camera.fps").real()
        self.mT = 1e3 / (fps if fps >= 1 else 30)

        self.mImageWidth = int(fSettings.getNode("Camera.width").real())
        self.mImageHeight = int(fSettings.getNode("Camera.height").real())
        if self.mImageWidth < 1 or self.mImageHeight < 1:
            self.mImageWidth = 640
            self.mImageHeight = 480

        self.mViewpointX = fSettings.getNode("Viewer.ViewpointX").real()
        self.mViewpointY = fSettings.getNode("Viewer.ViewpointY").real()
        self.mViewpointZ = fSettings.getNode("Viewer.ViewpointZ").real()
        self.mViewpointF = fSettings.getNode("Viewer.ViewpointF").real()

        fSettings.release()
