class MapDrawer:
    def __init__(self, pMap, fSettings):

        self.mKeyFrameSize = fSettings["Viewer.KeyFrameSize"]
        self.mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"]
        self.mGraphLineWidth = fSettings["Viewer.GraphLineWidth"]
        self.mPointSize = fSettings["Viewer.PointSize"]
        self.mCameraSize = fSettings["Viewer.CameraSize"]
        self.mCameraLineWidth = fSettings["Viewer.CameraLineWidth"]
