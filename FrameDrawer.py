import numpy as np
import threading
import cv2

class FrameDrawer:
    def __init__(self, pMap):

        self.mMutex = threading.Lock()
        self.mpMap = pMap
        self.mstate = "SYSTEM_NOT_READY"
        self.mIm = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mvCurrentKeys = []
        self.mvIniKeys = []
        self.mvIniMatches = []
        self.mvbVO = []
        self.mvbMap = []
        self.mnTracked = 0
        self.mnTrackedVO = 0
        self.mbOnlyTracking = False

    def draw_frame(self):
        im = None
        vIniKeys = []
        vMatches = []
        vCurrentKeys = []
        vbVO = []
        vbMap = []
        state = None

        with self.mMutex:
            state = self.mstate
            if self.mstate == "SYSTEM_NOT_READY":
                self.mstate = "NO_IMAGES_YET"

            if self.mIm is not None:
                im = self.mIm.copy()

            if self.mstate == "NOT_INITIALIZED":
                vCurrentKeys = self.mvCurrentKeys
                vIniKeys = self.mvIniKeys
                vMatches = self.mvIniMatches

            elif self.mstate == "OK":
                vCurrentKeys = self.mvCurrentKeys
                vbVO = self.mvbVO
                vbMap = self.mvbMap

            elif self.mstate == "LOST":
                vCurrentKeys = self.mvCurrentKeys

        if im is None:
            return None

        if len(im.shape) < 3 or im.shape[2] != 3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        if state == "NOT_INITIALIZED":
            for i, match in enumerate(vMatches):
                if match >= 0:
                    pt1 = tuple(map(int, vIniKeys[i].pt))
                    pt2 = tuple(map(int, vCurrentKeys[match].pt))
                    cv2.line(im, pt1, pt2, (0, 255, 0))
        elif state == "OK":
            self.mnTracked = 0
            self.mnTrackedVO = 0
            r = 5
            for i, keypoint in enumerate(vCurrentKeys):
                if vbVO[i] or vbMap[i]:
                    pt1 = (int(keypoint.pt[0] - r), int(keypoint.pt[1] - r))
                    pt2 = (int(keypoint.pt[0] + r), int(keypoint.pt[1] + r))
                    center = tuple(map(int, keypoint.pt))

                    if vbMap[i]:
                        cv2.rectangle(im, pt1, pt2, (0, 255, 0))
                        cv2.circle(im, center, 2, (0, 255, 0), -1)
                        self.mnTracked += 1
                    else:
                        cv2.rectangle(im, pt1, pt2, (255, 0, 0))
                        cv2.circle(im, center, 2, (255, 0, 0), -1)
                        self.mnTrackedVO += 1

        imWithInfo = self.draw_text_info(im, state)
        return imWithInfo

    def draw_text_info(self, im, nstate):
        s = ""
        if nstate == "NO_IMAGES_YET":
            s = " WAITING FOR IMAGES"
        elif nstate == "NOT_INITIALIZED":
            s = " TRYING TO INITIALIZE "
        elif nstate == "OK":
            if not self.mbOnlyTracking:
                s = "SLAM MODE |  "
            else:
                s = "LOCALIZATION | "
            nKFs = self.mpMap.key_frames_in_map()
            nMPs = self.mpMap.map_points_in_map()
            s += f"KFs: {nKFs}, MPs: {nMPs}, Matches: {self.mnTracked}"
            if self.mnTrackedVO > 0:
                s += f", + VO matches: {self.mnTrackedVO}"
        elif nstate == "LOST":
            s = " TRACK LOST. TRYING TO RELOCALIZE "
        elif nstate == "SYSTEM_NOT_READY":
            s = " LOADING ORB VOCABULARY. PLEASE WAIT..."

        baseline = 0
        textSize = cv2.getTextSize(s, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

        imText = np.zeros((im.shape[0] + textSize[1] + 10, im.shape[1], im.shape[2]), dtype=im.dtype)
        imText[:im.shape[0], :im.shape[1]] = im

        imText[im.shape[0]:] = 0

        cv2.putText(imText, s, (5, imText.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        return imText

    def update(self, pTracker):
        with self.mMutex:
            self.mIm = pTracker.mImGray.copy()

            self.mvCurrentKeys = pTracker.mCurrentFrame.mvKeys
            N = len(self.mvCurrentKeys)
            self.mvbVO = [False] * N
            self.mvbMap = [False] * N
            self.mbOnlyTracking = pTracker.mbOnlyTracking

            if pTracker.mLastProcessedState == "NOT_INITIALIZED":
                self.mvIniKeys = pTracker.mCurrentFrame.mvKeys

            elif pTracker.mLastProcessedState == "OK":
                for i in range(pTracker.mCurrentFrame.N):
                    pMP = pTracker.mCurrentFrame.mvpMapPoints[i]
                    if pMP:
                        if not pTracker.mCurrentFrame.mvbOutlier[i]:
                            if len(pMP.get_observations()) > 0:
                                self.mvbMap[i] = True
                            else:
                                self.mvbVO[i] = True

            self.mstate = pTracker.mLastProcessedState
