import numpy as np

class FrameDrawer:
    def __init__(self, pMap):
        self.mpMap = pMap
        self.mState = "SYSTEM_NOT_READY"
        self.mIm = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mvCurrentKeys = []  # Current frame keypoints
        self.mvIniKeys = []  # Initialization keypoints
        self.mvIniMatches = []  # Matches with initialization keypoints
        self.mvbVO = []  # Tracked visual odometry points
        self.mvbMap = []  # Tracked map points
        self.mnTracked = 0  # Count of tracked map points
        self.mnTrackedVO = 0  # Count of tracked visual odometry points
        self.mMutex = threading.Lock()  # Mutex for thread safety
        self.mbOnlyTracking = False  # Mode: Only tracking or SLAM

    def DrawFrame(self):
        """
        Draws the current frame with keypoints, matches, and tracking information.
        """
        im = None
        vIniKeys = []  # Initialization keypoints
        vMatches = []  # Matches with initialization keypoints
        vCurrentKeys = []  # Current frame keypoints
        vbVO = []  # Tracked visual odometry points
        vbMap = []  # Tracked map points
        state = None  # Tracking state

        # Copy variables within scoped mutex
        with self.mMutex:
            state = self.mState
            if self.mState == "SYSTEM_NOT_READY":
                self.mState = "NO_IMAGES_YET"

            if self.mIm is not None:
                im = self.mIm.copy()

            if self.mState == "NOT_INITIALIZED":
                vCurrentKeys = self.mvCurrentKeys
                vIniKeys = self.mvIniKeys
                vMatches = self.mvIniMatches
            elif self.mState == "OK":
                vCurrentKeys = self.mvCurrentKeys
                vbVO = self.mvbVO
                vbMap = self.mvbMap
            elif self.mState == "LOST":
                vCurrentKeys = self.mvCurrentKeys

        if im is None:
            return None

        # Convert to color image if grayscale
        if len(im.shape) < 3 or im.shape[2] != 3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        # Draw
        if state == "NOT_INITIALIZED":  # INITIALIZING
            for i, match in enumerate(vMatches):
                if match >= 0:
                    pt1 = tuple(map(int, vIniKeys[i].pt))
                    pt2 = tuple(map(int, vCurrentKeys[match].pt))
                    cv2.line(im, pt1, pt2, (0, 255, 0))  # Green line
        elif state == "OK":  # TRACKING
            self.mnTracked = 0
            self.mnTrackedVO = 0
            r = 5
            for i, keypoint in enumerate(vCurrentKeys):
                if vbVO[i] or vbMap[i]:
                    pt1 = (int(keypoint.pt[0] - r), int(keypoint.pt[1] - r))
                    pt2 = (int(keypoint.pt[0] + r), int(keypoint.pt[1] + r))
                    center = tuple(map(int, keypoint.pt))

                    if vbMap[i]:  # Match to a MapPoint
                        cv2.rectangle(im, pt1, pt2, (0, 255, 0))  # Green rectangle
                        cv2.circle(im, center, 2, (0, 255, 0), -1)  # Green circle
                        self.mnTracked += 1
                    else:  # Match to a visual odometry MapPoint
                        cv2.rectangle(im, pt1, pt2, (255, 0, 0))  # Blue rectangle
                        cv2.circle(im, center, 2, (255, 0, 0), -1)  # Blue circle
                        self.mnTrackedVO += 1

        # Add tracking information
        imWithInfo = self.DrawTextInfo(im, state)
        return imWithInfo

    def DrawTextInfo(self, im, nState):
        """
        Adds text information about the current tracking state to the image.

        Parameters:
        - im: The input image (numpy array).
        - nState: The tracking state.

        Returns:
        - imText: The output image with text information added.
        """
        s = ""

        if nState == "NO_IMAGES_YET":
            s = " WAITING FOR IMAGES"
        elif nState == "NOT_INITIALIZED":
            s = " TRYING TO INITIALIZE "
        elif nState == "OK":
            if not self.mbOnlyTracking:
                s = "SLAM MODE |  "
            else:
                s = "LOCALIZATION | "
            nKFs = self.mpMap.KeyFramesInMap()
            nMPs = self.mpMap.MapPointsInMap()
            s += f"KFs: {nKFs}, MPs: {nMPs}, Matches: {self.mnTracked}"
            if self.mnTrackedVO > 0:
                s += f", + VO matches: {self.mnTrackedVO}"
        elif nState == "LOST":
            s = " TRACK LOST. TRYING TO RELOCALIZE "
        elif nState == "SYSTEM_NOT_READY":
            s = " LOADING ORB VOCABULARY. PLEASE WAIT..."

        # Calculate text size
        baseline = 0
        textSize = cv2.getTextSize(s, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

        # Create a new image with space for text
        imText = np.zeros((im.shape[0] + textSize[1] + 10, im.shape[1], im.shape[2]), dtype=im.dtype)
        imText[:im.shape[0], :im.shape[1]] = im  # Copy original image

        # Add a black bar for text at the bottom
        imText[im.shape[0]:] = 0

        # Add text to the image
        cv2.putText(imText, s, (5, imText.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        return imText



    def Update(self, pTracker):
        """
        Update the frame drawer with information from the tracker.

        Parameters:
        - pTracker: Tracker object providing the latest frame and state.
        """
        with self.mMutex:
            # Copy grayscale image
            self.mIm = pTracker.mImGray.copy()  # Assuming pTracker.mImGray is a numpy array

            # Copy current keypoints and initialize flags
            self.mvCurrentKeys = pTracker.mCurrentFrame.mvKeys
            N = len(self.mvCurrentKeys)
            self.mvbVO = [False] * N
            self.mvbMap = [False] * N
            self.mbOnlyTracking = pTracker.mbOnlyTracking

            # Handle states
            if pTracker.mLastProcessedState == "NOT_INITIALIZED":
                # Save initialization data
                self.mvIniKeys = pTracker.mInitialFrame.mvKeys
                self.mvIniMatches = pTracker.mvIniMatches
            elif pTracker.mLastProcessedState == "OK":
                # Mark map points and visual odometry points
                for i in range(N):
                    pMP = pTracker.mCurrentFrame.mvpMapPoints[i]
                    if pMP:
                        if not pTracker.mCurrentFrame.mvbOutlier[i]:  # Not an outlier
                            if pMP.Observations() > 0:
                                self.mvbMap[i] = True  # Map point
                            else:
                                self.mvbVO[i] = True  # Visual odometry point

            # Update tracking state
            self.mState = pTracker.mLastProcessedState
