class MapDrawer:
    def __init__(self, pMap, strSettingPath):
        """
        Initializes the MapDrawer.

        Args:
            pMap (Map): The map object.
            strSettingPath (str): Path to the settings file.
        """
        self.mpMap = pMap

        fSettings = cv2.FileStorage(strSettingPath, cv2.FILE_STORAGE_READ)

        self.mKeyFrameSize = fSettings.getNode("Viewer.KeyFrameSize").real()
        self.mKeyFrameLineWidth = fSettings.getNode("Viewer.KeyFrameLineWidth").real()
        self.mGraphLineWidth = fSettings.getNode("Viewer.GraphLineWidth").real()
        self.mPointSize = fSettings.getNode("Viewer.PointSize").real()
        self.mCameraSize = fSettings.getNode("Viewer.CameraSize").real()
        self.mCameraLineWidth = fSettings.getNode("Viewer.CameraLineWidth").real()

        fSettings.release()
