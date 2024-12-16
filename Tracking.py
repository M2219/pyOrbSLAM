import threading
import numpy as np
import cv2

from copy import deepcopy

from Frame import Frame
from KeyFrame import KeyFrame
from MapPoint import MapPoint
from ORBExtractor import ORBExtractor
from ORBMatcher import ORBMatcher

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
        self.mlRelativeFramePoses = []
        self.mlpReferences = []
        self.mlFrameTimes = []
        self.mlbLost = []
        self.mVelocity = None

        self.fx = fSettings["Camera.fx"]
        self.fy = fSettings["Camera.fy"]
        self.cx = fSettings["Camera.cx"]
        self.cy = fSettings["Camera.cy"]

        self.invfx = 1.0 / self.fx
        self.invfy = 1.0 / self.fy

        self.mK = np.eye(3, dtype=np.float32)
        self.mK[0, 0] = self.fx
        self.mK[1, 1] = self.fy
        self.mK[0, 2] = self.cx
        self.mK[1, 2] = self.cy

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

        self.mpORBExtractorLeft = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        self.mpORBExtractorRight = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        self.mThDepth = self.mbf * fSettings["ThDepth"] / self.fx


    def grab_image_stereo(self, mImGray, imGrayRight, timestamp):
        ##############
        # just for the first image should be transfer to stereo and done for the first loop than frame_args passes to the
        ##############

        FRAME_GRID_ROWS = 48
        FRAME_GRID_COLS = 64

        self.mImGray = mImGray
        self.imGrayRight = imGrayRight

        mnMinX, mnMaxX, mnMinY, mnMaxY = self.compute_image_bounds(mImGray, self.mK, self.mDistCoef)


        mfGridElementWidthInv = float(FRAME_GRID_COLS) / (mnMaxX - mnMinX)
        mfGridElementHeightInv = float(FRAME_GRID_ROWS) / (mnMaxY - mnMinY)
        ########################

        self.frame_args = [self.fx, self.fy, self.cx, self.cy, self.invfx, self.invfy,
          mfGridElementWidthInv, mfGridElementHeightInv, mnMinX, mnMaxX, mnMinY, mnMaxY, FRAME_GRID_ROWS, FRAME_GRID_COLS]


        if len(mImGray.shape) == 2:
            print("Images are grayscale!")
        else:
            print("Convert images to grayscale!")
            exit(-1)


        self.mCurrentFrame = Frame(self.mImGray, self.imGrayRight, timestamp, self.mpORBExtractorLeft, self.mpORBExtractorRight, self.mpORBVocabulary,
                        self.mK, self.mDistCoef, self.mbf, self.mThDepth, self.frame_args)

        # Run tracking
        self.track()

        # Return the pose of the current frame
        return self.mCurrentFrame.mTcw.copy()

    def compute_image_bounds(self, imLeft, mK, mDistCoef):

        if mDistCoef[0][0] != 0.0:
            mat = np.zeros((4, 2), dtype=np.float32)
            mat[0, 0] = 0.0
            mat[0, 1] = 0.0
            mat[1, 0] = imLeft.shape[1]
            mat[1, 1] = 0.0
            mat[2, 0] = 0.0
            mat[2, 1] = imLeft.shape[0]
            mat[3, 0] = imLeft.shape[1]
            mat[3, 1] = imLeft.shape[0]

            mat = mat.reshape(-1, 1, 2)
            mat = cv2.undistortPoints(mat, mK, mDistCoef, None, mK)
            mat = mat.reshape(-1, 2)

            mnMinX = min(mat[0, 0], mat[2, 0])
            mnMaxX = max(mat[1, 0], mat[3, 0])
            mnMinY = min(mat[0, 1], mat[1, 1])
            mnMaxY = max(mat[2, 1], mat[3, 1])
        else:
            mnMinX = 0.0
            mnMaxX = imLeft.shape[1]
            mnMinY = 0.0
            mnMaxY = imLeft.shape[0]

        return mnMinX, mnMaxX, mnMinY, mnMaxY

    def track(self):
        """
        Main tracking function
        """
        if self.mState == "NO_IMAGES_YET":
            self.mState = "NOT_INITIALIZED"

        self.mLastProcessedState = self.mState

        # Ensure thread-safe access to the map
        with self.mpMap.mMutexMapUpdate:
            if self.mState == "NOT_INITIALIZED":
                self.stereo_initialization()
                self.mpFrameDrawer.update(self)

                if self.mState != "OK":
                    return
            else:
                # System is initialized; track the frame
                bOK = False

                if not self.mbOnlyTracking:
                    if self.mState == "OK":
                        # Check replaced MapPoints in the last frame
                        self.check_replaced_in_last_frame()

                        if not self.mVelocity or self.mCurrentFrame.mnId < self.mnLastRelocFrameId + 2:
                            bOK = self.track_reference_key_frame()
                        else:
                            bOK = self.track_with_motion_model()
                            if not bOK:
                                bOK = self.track_reference_key_frame()
                    else:
                        bOK = self.relocalization()
                else:
                    # Only tracking mode
                    if self.mState == "LOST":
                        bOK = self.relocalization()
                    else:
                        if not self.mbVO:
                            if self.mVelocity:
                                bOK = self.track_with_motion_model()
                            else:
                                bOK = self.track_reference_key_frame()
                        else:
                            # Handle visual odometry mode
                            bOKMM, bOKReloc = False, False
                            vpMPsMM, vbOutMM, TcwMM = [], [], None

                            if self.mVelocity:
                                bOKMM = self.track_with_motion_model()
                                vpMPsMM = self.mCurrentFrame.mvpMapPoints.copy()
                                vbOutMM = self.mCurrentFrame.mvbOutlier.copy()
                                TcwMM = self.mCurrentFrame.mTcw.copy()

                            bOKReloc = self.relocalization()

                            if bOKMM and not bOKReloc:
                                self.mCurrentFrame.set_pose(TcwMM)
                                self.mCurrentFrame.mvpMapPoints = vpMPsMM
                                self.mCurrentFrame.mvbOutlier = vbOutMM

                                if self.mbVO:
                                    for i in self.indx:
                                        if (
                                            self.mCurrentFrame.mvpMapPoints[i]
                                            and not self.mCurrentFrame.mvbOutlier[i]
                                        ):
                                            self.mCurrentFrame.mvpMapPoints[i].increase_found()
                            elif bOKReloc:
                                self.mbVO = False

                            bOK = bOKReloc or bOKMM

                self.mCurrentFrame.mpReferenceKF = self.mpReferenceKF

                # Track the local map if the pose is estimated
                if not self.mbOnlyTracking:
                    if bOK:
                        bOK = self.track_local_map()
                else:
                    if bOK and not self.mbVO:
                        bOK = self.track_local_map()

                # Update tracking state
                self.mState = "OK" if bOK else "LOST"

                # Update drawer
                self.mpFrameDrawer.update(self)

                if bOK:
                    # Update motion model
                    if self.mLastFrame and self.mLastFrame.mTcw is not None:
                        LastTwc = np.eye(4, dtype=np.float32)
                        self.mLastFrame.get_rotation_inverse().copy_to(LastTwc[:3, :3])
                        self.mLastFrame.get_camera_center().copy_to(LastTwc[:3, 3])
                        self.mVelocity = self.mCurrentFrame.mTcw @ LastTwc
                    else:
                        self.mVelocity = None

                    self.mpMapDrawer.set_current_camera_pose(self.mCurrentFrame.mTcw)

                    # Clean VO matches
                    for i in self.indx:
                        pMP = self.mCurrentFrame.mvpMapPoints[i]
                        if pMP and pMP.observations() < 1:
                            self.mCurrentFrame.mvbOutlier[i] = False
                            self.mCurrentFrame.mvpMapPoints[i] = None

                    # Delete temporary MapPoints
                    for pMP in self.mlpTemporalPoints:
                        del pMP
                    self.mlpTemporalPoints.clear()

                    # Check if we need to insert a new keyframe
                    if self.need_new_keyframe():
                        self.create_new_keyframe()

                    # Remove high-innovation points
                    for i in self.indx:
                        if self.mCurrentFrame.mvpMapPoints[i] and self.mCurrentFrame.mvbOutlier[i]:
                            self.mCurrentFrame.mvpMapPoints[i] = None

                # Reset if tracking is lost soon after initialization
                if self.mState == "LOST":
                    if self.mpMap.keyframes_in_map() <= 5:
                        print("Track lost soon after initialization, resetting...")
                        self.mpSystem.reset()
                        return

                if not self.mCurrentFrame.mpReferenceKF:
                    self.mCurrentFrame.mpReferenceKF = self.mpReferenceKF

                self.mLastFrame = Frame(self.mCurrentFrame)

        # Store frame pose information for trajectory retrieval
        if self.mCurrentFrame.mTcw is not None:
            Tcr = self.mCurrentFrame.mTcw @ self.mCurrentFrame.mpReferenceKF.get_pose_inverse()
            self.mlRelativeFramePoses.append(Tcr)
            self.mlpReferences.append(self.mpReferenceKF)
            self.mlFrameTimes.append(self.mCurrentFrame.mTimeStamp)
            self.mlbLost.append(self.mState == "LOST")
        else:
            # If tracking is lost, replicate the last known values
            self.mlRelativeFramePoses.append(self.mlRelativeFramePoses[-1])
            self.mlpReferences.append(self.mlpReferences[-1])
            self.mlFrameTimes.append(self.mlFrameTimes[-1])
            self.mlbLost.append(self.mState == "LOST")
        print("tracking first loop")

    def stereo_initialization(self):

        if self.mCurrentFrame.N > 500:
            self.mCurrentFrame.set_pose(np.eye(4, 4, dtype=np.float32))
            self.mCurrentFrame.compute_BoW()
            pKFini = KeyFrame(self.mCurrentFrame, self.mpMap, self.mpKeyFrameDB)

            # Insert KeyFrame into the map
            self.mpMap.add_key_frame(pKFini)
            self.indx = []
            # Create MapPoints and associate them to the KeyFrame
            for i in range(self.mCurrentFrame.N):
                z = self.mCurrentFrame.mvDepth[i]
                if z > 0:
                    x3D = self.mCurrentFrame.unproject_stereo(i)
                    pNewMP = MapPoint(x3D, pKFini, self.mpMap)
                    pNewMP.add_observation(pKFini, i)
                    pKFini.add_map_point(pNewMP, i)
                    pNewMP.compute_distinctive_descriptors()
                    pNewMP.update_normal_and_depth()
                    self.mpMap.add_map_point(pNewMP)
                    self.mCurrentFrame.mvpMapPoints[i] = pNewMP
                    self.indx.append(i)


            print(f"New map created with {self.mpMap.map_points_in_map()} points")

            # Insert KeyFrame into the local mapper
            with self.ss["pKF_ss_lock"]: # check the speed without this lock
                self.ss["pKF_ss"] = pKFini # transfer insert to the while

            # Update frame and keyframe references
            self.mLastFrame = self.mCurrentFrame.copy(self.mCurrentFrame)
            self.mnLastKeyFrameId = self.mCurrentFrame.mnId
            self.mpLastKeyFrame = pKFini
            self.mvpLocalKeyFrames = [pKFini]
            self.mvpLocalMapPoints = self.mpMap.get_all_map_points()
            self.mpReferenceKF = pKFini
            self.mCurrentFrame.mpReferenceKF = pKFini

            self.mpMap.set_reference_map_points(self.mvpLocalMapPoints)
            self.mpMap.mvpKeyFrameOrigins.append(pKFini)

            # Update the map drawer with the current camera pose
            self.mpMapDrawer.set_current_camera_pose(self.mCurrentFrame.mTcw)

            # Set state to OK
            self.mState = "OK"

    def check_replaced_in_last_frame(self):
        """
        Update replaced map points in the last frame with their replacements.
        """
        for i in self.indx:
            pMP = self.mLastFrame.mvpMapPoints[i]
            pRep = pMP.get_replaced() # print and check
            if pRep is not None:
                self.mLastFrame.mvpMapPoints[i] = pRep

    def track_reference_key_frame(self):
        """
        Track the reference keyframe and update the current frame pose.

        Returns:
        - True if successful, False otherwise.
        """
        # Compute Bag of Words vector
        self.mCurrentFrame.compute_BoW()

        # Perform ORB matching with the reference keyframe
        # If enough matches are found, setup a PnP solver
        matcher = ORBMatcher(0.7, True)
        vp_map_point_matches = []

        nmatches = matcher.search_by_BoW(self.mpReferenceKF, self.mCurrentFrame, vp_map_point_matches)

        if nmatches < 15:
            return False

        self.mCurrentFrame.mvpMapPoints = vp_map_point_matches
        self.mCurrentFrame.set_pose(self.mLastFrame.mTcw)

        # Optimize the pose
        Optimizer.pose_optimization(self.mCurrentFrame)

        # Discard outliers
        nmatches_map = 0
        for i in self.indx:
            if self.mCurrentFrame.mvpMapPoints[i]:
                if self.mCurrentFrame.mvbOutlier[i]:
                    pMP = self.mCurrentFrame.mvpMapPoints[i]
                    self.mCurrentFrame.mvpMapPoints[i] = None
                    self.mCurrentFrame.mvbOutlier[i] = False
                    pMP.mbTrackInView = False
                    pMP.mnLastFrameSeen = self.mCurrentFrame.mnId
                    nmatches -= 1
                elif self.mCurrentFrame.mvpMapPoints[i].observations() > 0:
                    nmatches_map += 1

        return nmatches_map >= 10

if __name__ == "__main__":

    import yaml
    from pyDBoW.TemplatedVocabulary import TemplatedVocabulary
    from ORBExtractor import ORBExtractor
    from stereo_kitti import LoadImages
    from KeyFrameDatabase import KeyFrameDatabase
    from Map import Map
    from FrameDrawer import FrameDrawer
    from MapDrawer import MapDrawer

    vocabulary = TemplatedVocabulary(k=5, L=3, weighting="TF_IDF", scoring="L1_NORM")
    vocabulary.load_from_text_file("./Vocabulary/ORBvoc.txt")

    with open("configs/KITTI00-02.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    leftImages, rightImages, timeStamps = LoadImages("00")
    nImages = len(leftImages)

    ss = {
           "pKF_ss": None, "pKF_ss_lock" : threading.Lock()
         }


    mpKeyFrameDatabase = KeyFrameDatabase(vocabulary)

    mpMap = Map()

    mpFrameDrawer = FrameDrawer(mpMap)
    mpMapDrawer = MapDrawer(mpMap, cfg)

    mpTracker = Tracking(False, vocabulary, mpFrameDrawer, mpMapDrawer,
                                  mpMap, mpKeyFrameDatabase, cfg, sensor="Stereo", ss=ss)



    for i in range(2):
        mleft = cv2.imread(leftImages[i], cv2.IMREAD_GRAYSCALE)
        mright = cv2.imread(rightImages[i], cv2.IMREAD_GRAYSCALE)
        timestamp = float(timeStamps[i])

        Twc = mpTracker.grab_image_stereo(mleft, mright, timestamp)
        print(Twc)

