import threading
import time
import sys

import numpy as np
import cv2

from copy import deepcopy
from ordered_set import OrderedSet

from Frame import Frame
from KeyFrame import KeyFrame
from ORBMatcher import ORBMatcher
from Optimizer import Optimizer
from MapPoint import MapPoint
from PnPsolver import PnPsolver

sys.path.append("./pyORBExtractor/lib/")
from pyORBExtractor import ORBextractor


class Tracking:
    def __init__(self, pSys, pVoc, pFrameDrawer, pMapDrawer, pMap, pKFDB, fSettings, sensor):

        self.mState = "NO_IMAGES_YET"
        self.mSensor = sensor
        self.mbOnlyTracking = False
        self.mbVO = False
        self.mpORBVocabulary = pVoc
        self.mpKeyFrameDB = pKFDB
        self.mpInitializer = None
        self.mpSystem = pSys
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

        self.mpORBExtractorLeft = ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        self.mpORBExtractorRight = ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
        self.mThDepth = self.mbf * fSettings["ThDepth"] / self.fx

        self.mlpTemporalPoints = []

        self.optimizer = Optimizer()

    @property
    def mpLocalMapper(self):
        return self.mpSystem.mpLocalMapper

    @property
    def mpLoopCloser(self):
        return self.mpSystem.mpLoopCloser

    @property
    def mpViewer(self):
        return self.mpSystem.mpViewer

    def grab_image_stereo(self, mImGray, imGrayRight, timestamp, i):

        FRAME_GRID_ROWS = 48
        FRAME_GRID_COLS = 64

        self.mImGray = mImGray
        self.imGrayRight = imGrayRight

        mnMinX, mnMaxX, mnMinY, mnMaxY = self.compute_image_bounds(mImGray, self.mK, self.mDistCoef)


        mfGridElementWidthInv = float(FRAME_GRID_COLS) / (mnMaxX - mnMinX)
        mfGridElementHeightInv = float(FRAME_GRID_ROWS) / (mnMaxY - mnMinY)

        self.frame_args = [self.fx, self.fy, self.cx, self.cy, self.invfx, self.invfy,
          mfGridElementWidthInv, mfGridElementHeightInv, mnMinX, mnMaxX, mnMinY, mnMaxY, FRAME_GRID_ROWS, FRAME_GRID_COLS]

        self.mCurrentFrame = Frame(self.mImGray, self.imGrayRight, timestamp, self.mpORBExtractorLeft, self.mpORBExtractorRight, self.mpORBVocabulary,
                        self.mK, self.mDistCoef, self.mbf, self.mThDepth, self.frame_args)

        self.track(i)

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

    def track(self, imm):
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

                        if  (self.mVelocity is None) or (self.mCurrentFrame.mnId < self.mnLastRelocFrameId + 2):
                            print("-------------------------------------------> 1")
                            bOK = self.track_reference_key_frame()

                        else:

                            print("--------------------------------------------- > 2")
                            bOK = self.track_with_motion_model()
                            if not bOK:
                                print("-------------------------------------------------> 3")
                                bOK = self.track_reference_key_frame()
                    else:
                        bOK = self.relocalization()
                else:
                    print("====================> in else")
                    # Only tracking mode
                    if self.mState == "LOST":
                        bOK = self.relocalization()
                    else:
                        if not self.mbVO:
                            if self.mVelocity is not None:
                                bOK = self.track_with_motion_model()
                            else:
                                bOK = self.track_reference_key_frame()
                        else:
                            # Handle visual odometry mode
                            bOKMM, bOKReloc = False, False
                            vpMPsMM, vbOutMM, TcwMM = [], [], None

                            if self.mVelocity is not None:
                                print("empty list in dictionary !")
                                bOKMM = self.track_with_motion_model()
                                vpMPsMM = self.mCurrentFrame.mvpMapPoints
                                vbOutMM = self.mCurrentFrame.mvbOutlier
                                TcwMM = self.mCurrentFrame.mTcw.copy()

                            bOKReloc = self.relocalization()

                            if bOKMM and not bOKReloc:
                                print("empty list in dictionary !")
                                self.mCurrentFrame.set_pose(TcwMM)
                                self.mCurrentFrame.mvpMapPoints = vpMPsMM
                                self.mCurrentFrame.mvbOutlier = vbOutMM

                                if self.mbVO:
                                    for i in range(self.mCurrentFrame.N):
                                        if (self.mCurrentFrame.mvpMapPoints[i] and not self.mCurrentFrame.mvbOutlier[i]):
                                            pself.mCurrentFrame.mvpMapPoints[i].increase_found()
                            elif bOKReloc:
                                self.mbVO = False

                            bOK = bOKReloc or bOKMM

                self.mCurrentFrame.mpReferenceKF = self.mpReferenceKF

                # Track the local map if the pose is estimated
                if not self.mbOnlyTracking:
                    if bOK:
                        bOK = self.track_local_map()
                        #print("current local map -->", self.mCurrentFrame.mTcw)
                else:
                    if bOK and not self.mbVO:
                        bOK = self.track_local_map()
                        #print("update 2 -->", self.mCurrentFrame.mTcw)


                # Update tracking state
                #self.mState = "OK" if bOK else "LOST"
                #if imm > 10:
                #    self.mState = "LOST"

                # Update drawer
                self.mpFrameDrawer.update(self)

                if bOK:
                    # Update motion model
                    if self.mLastFrame and self.mLastFrame.mTcw is not None:
                        LastTwc = np.concatenate((self.mLastFrame.get_rotation_inverse(), self.mLastFrame.get_camera_center()), axis=1)
                        LastTwc = np.concatenate((LastTwc, np.array([[0, 0, 0, 1]])), axis = 0)

                        self.mVelocity = self.mCurrentFrame.mTcw @ LastTwc

                    else:
                        self.mVelocity = None

                    self.mpMapDrawer.set_current_camera_pose(self.mCurrentFrame.mTcw)

                    # Clean VO matches
                    for i in range(self.mCurrentFrame.N):
                        pMP = self.mCurrentFrame.mvpMapPoints[i]
                        if pMP:
                            if pMP.observations() < 1:
                                self.mCurrentFrame.mvbOutlier[i] = False
                                self.mCurrentFrame.mvpMapPoints[i] = None


                    # Delete temporary MapPoints
                    for pMP in self.mlpTemporalPoints: # check if it works
                        for i in range(self.mCurrentFrame.N):
                            if self.mCurrentFrame.mvpMapPoints[i] == pMP:
                                self.mCurrentFrame.mvbOutlier[i] = False
                                self.mCurrentFrame.mvpMapPoints[i] = None

                    self.mlpTemporalPoints.clear()


                    # Check if we need to insert a new keyframe
                    if self.need_new_key_frame():
                        self.create_new_key_frame()

                    # Remove high-innovation points
                    for i in range(self.mCurrentFrame.N):
                        if self.mCurrentFrame.mvbOutlier[i] and self.mCurrentFrame.mvpMapPoints[i]:
                             self.mCurrentFrame.mvpMapPoints[i] = None

                # Reset if tracking is lost soon after initialization
                if self.mState == "LOST":
                    if self.mpMap.key_frames_in_map() <= 5:
                        print("Track lost soon after initialization, resetting...")
                        self.mpSystem.reset()
                        return

                if not self.mCurrentFrame.mpReferenceKF:
                    self.mCurrentFrame.mpReferenceKF = self.mpReferenceKF

                #print("blen mvpMapPoints", len(self.mCurrentFrame.mvpMapPoints))
                #print("blen mvbOutlier", len(self.mCurrentFrame.mvbOutlier))

                self.mLastFrame = self.mCurrentFrame.copy(self.mCurrentFrame)
                #print("last after local map -->", self.mLastFrame.mTcw)
                #print("alen mvpMapPoints", len(self.mLastFrame.mvpMapPoints))
                #print("alen mvbOutlier", len(self.mLastFrame.mvbOutlier))


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

    def stereo_initialization(self):

        if self.mCurrentFrame.N > 500:
            self.mCurrentFrame.set_pose(np.eye(4, 4, dtype=np.float32))
            self.mCurrentFrame.compute_BoW()
            pKFini = KeyFrame(self.mCurrentFrame, self.mpMap, self.mpKeyFrameDB)

            # Insert KeyFrame into the map
            self.mpMap.add_key_frame(pKFini)
            # Create MapPoints and associate them to the KeyFrame
            for i in range(self.mCurrentFrame.N):
                z = self.mCurrentFrame.mvDepth[i]
                if z > 0:
                    x3D = self.mCurrentFrame.unproject_stereo(i)
                    self.pNewMP = MapPoint(x3D, pKFini, self.mpMap, idxF=i, kframe_bool=True)
                    self.pNewMP.add_observation(pKFini, i)
                    pKFini.add_map_point(self.pNewMP, i)
                    self.pNewMP.compute_distinctive_descriptors()
                    self.pNewMP.update_normal_and_depth()
                    self.mpMap.add_map_point(self.pNewMP)
                    self.mCurrentFrame.mvpMapPoints[i] = self.pNewMP
                    self.mCurrentFrame.mvbOutlier[i] = False

            print(f"New map created with {self.mpMap.map_points_in_map()} points")

            # Insert KeyFrame into the local mapper
            #with self.mMutexNewKFs: # check the speed without this lock
            self.mpLocalMapper.insert_key_frame(pKFini) # transfer insert to the while

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
        for i in range(self.mLastFrame.N):
            pMP = self.mLastFrame.mvpMapPoints[i]
            if pMP:
                pRep = pMP.get_replaced() # print and check
                if pRep:
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
        nmatches, vpMapPointMatches = matcher.search_by_BoW_kf_f(self.mpReferenceKF, self.mCurrentFrame)

        if nmatches < 15:
            return False

        self.mCurrentFrame.mvpMapPoints = vpMapPointMatches
        self.mCurrentFrame.set_pose(self.mLastFrame.mTcw)

        # Optimize the pose
        self.optimizer.pose_optimization(self.mCurrentFrame)

        #print("len mvpMapPoints in track", len(self.mCurrentFrame.mvpMapPoints))
        #print("len mvbOutlier in track", len(self.mCurrentFrame.mvbOutlier))

        #        self.mLastFrame = self.mCurrentFrame.copy(self.mCurrentFrame)
        #        #print("last after local map -->", self.mLastFrame.mTcw)
        #        print("len mvpMapPoints", len(self.mLastFrame.mvpMapPoints))
        #        print("len mvbOutlier", len(self.mLastFrame.mvbOutlier))

        # Discard outliers
        nmatches_map = 0
        for i in range(self.mCurrentFrame.N):
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

    def track_local_map(self):
        """
        Perform local map tracking by updating the local map, searching for local points,
        and optimizing the pose of the current frame.

        Returns:
            bool: True if tracking is successful, False otherwise.
        """
        self.update_local_map()
        self.search_local_points()

        self.optimizer.pose_optimization(self.mCurrentFrame)

        self.mnMatchesInliers = 0

        # Update MapPoint statistics
        for i in range(self.mCurrentFrame.N):
            if self.mCurrentFrame.mvpMapPoints[i]:
                if not self.mCurrentFrame.mvbOutlier[i]:
                    self.mCurrentFrame.mvpMapPoints[i].increase_found()
                    if not self.mbOnlyTracking:
                        if self.mCurrentFrame.mvpMapPoints[i].observations() > 0:
                            self.mnMatchesInliers += 1
                    else:
                        self.mnMatchesInliers += 1
                else:
                    self.mCurrentFrame.mvpMapPoints[i] = None

        # Decide if the tracking was successful
        # More restrictive if there was a recent relocalization
        if self.mCurrentFrame.mnId < self.mnLastRelocFrameId + self.mMaxFrames and self.mnMatchesInliers < 50:
            return False

        if self.mnMatchesInliers < 30:
            return False
        else:
            return True

    def update_local_map(self):
        """
        Update the local map by setting the reference map points for visualization
        and updating local keyframes and points.
        """
        # Set reference map points for visualization

        self.mpMap.set_reference_map_points(self.mvpLocalMapPoints)

        # Update local keyframes and points
        self.update_local_keyframes()
        self.update_local_points()


    def update_local_keyframes(self):
        """
        Update the local keyframes by voting for the keyframes that observe the map points
        in the current frame and identifying the keyframe that shares the most points.
        """
        # Map point voting for keyframes
        self.keyframeCounter = {}
        for i in range(self.mCurrentFrame.N):
            if self.mCurrentFrame.mvpMapPoints[i]:
                pMP = self.mCurrentFrame.mvpMapPoints[i]
                if not pMP.is_bad():
                    observations = pMP.get_observations()
                    for pKF, _ in observations.items():
                        if pKF not in self.keyframeCounter:
                            self.keyframeCounter[pKF] = 0
                        self.keyframeCounter[pKF] += 1
                else:
                    self.mCurrentFrame.mvpMapPoints[i] = None

        if not self.keyframeCounter:
            return

        # Find the keyframe with the most shared points
        max_votes = 0
        pKFmax = None

        self.mvpLocalKeyFrames.clear()

        # Include all keyframes observing a map point in the local map
        for pKF, count in self.keyframeCounter.items():
            if pKF.is_bad():
                continue

            if count > max_votes:
                max_votes = count
                pKFmax = pKF

            self.mvpLocalKeyFrames.append(pKF)
            pKF.mnTrackReferenceForFrame = self.mCurrentFrame.mnId

    def update_local_points(self):
        """
        Update the local map points by collecting map points from local keyframes.
        """
        self.mvpLocalMapPoints.clear()
        for pKF in self.mvpLocalKeyFrames:
            vpMPs = pKF.get_map_point_matches()
            for pMP in vpMPs:
                if not pMP:
                    continue
                if pMP.mnTrackReferenceForFrame == self.mCurrentFrame.mnId:
                    continue
                if not pMP.is_bad():
                    self.mvpLocalMapPoints.append(pMP)
                    pMP.mnTrackReferenceForFrame = self.mCurrentFrame.mnId


    def search_local_points(self):
        """
        Search for local map points and update their visibility and tracking status.
        """
        # Do not search map points already matched
        for i in range(self.mCurrentFrame.N):
            pMP = self.mCurrentFrame.mvpMapPoints[i]
            if pMP:
                if pMP.is_bad():
                    self.mCurrentFrame.mvpMapPoints[i] = None
                else:
                    pMP.increase_visible()
                    pMP.mnLastFrameSeen = self.mCurrentFrame.mnId
                    pMP.mbTrackInView = False

        nToMatch = 0
        # Project points in frame and check their visibility
        for pMP in self.mvpLocalMapPoints:
            if pMP.mnLastFrameSeen == self.mCurrentFrame.mnId:
                continue
            if pMP.is_bad():
                continue
            # Project (this fills MapPoint variables for matching)
            if self.mCurrentFrame.is_in_frustum(pMP, 0.5):
                pMP.increase_visible()
                nToMatch += 1

        if nToMatch > 0:
            matcher = ORBMatcher(0.8, True)
            th = 1
            # If the camera has been recently relocalized, perform a coarser search
            if self.mCurrentFrame.mnId < self.mnLastRelocFrameId + 2:
                th = 5
            matcher.search_by_projection_f_p(self.mCurrentFrame, self.mvpLocalMapPoints, th)

    def need_new_key_frame(self):
        """
        Determine if a new keyframe needs to be added to the map.

        Returns:
            bool: True if a new keyframe is needed, False otherwise.
        """

        if self.mbOnlyTracking:
            return False

        # If Local Mapping is frozen by a Loop Closure, do not insert keyframes
        if self.mpLocalMapper.is_stopped() or self.mpLocalMapper.stop_requested():
            return False


        nKFs = self.mpMap.key_frames_in_map()
        # Do not insert keyframes if not enough frames have passed since last relocalization
        if self.mCurrentFrame.mnId < self.mnLastRelocFrameId + self.mMaxFrames and nKFs > self.mMaxFrames:
            return False

        # Tracked MapPoints in the reference keyframe
        nMinObs = 3
        if nKFs <= 2:
            nMinObs = 2

        nRefMatches = self.mpReferenceKF.tracked_map_points(nMinObs)

        # Is Local Mapping accepting keyframes?
        bLocalMappingIdle = self.mpLocalMapper.accept_key_frames()


        # Check "close" points being tracked and potential new points
        nNonTrackedClose = 0
        nTrackedClose = 0
        for i in range(self.mCurrentFrame.N):
            if 0 < self.mCurrentFrame.mvDepth[i] < self.mThDepth:
                if self.mCurrentFrame.mvpMapPoints[i] and not self.mCurrentFrame.mvbOutlier[i]:
                    nTrackedClose += 1
                else:
                    nNonTrackedClose += 1

        bNeedToInsertClose = (nTrackedClose < 100) and (nNonTrackedClose > 70)

        # Thresholds
        thRefRatio = 0.75
        if nKFs < 2:
            thRefRatio = 0.4

        # Conditions
        c1a = self.mCurrentFrame.mnId >= self.mnLastKeyFrameId + self.mMaxFrames
        c1b = self.mCurrentFrame.mnId >= self.mnLastKeyFrameId + self.mMinFrames and bLocalMappingIdle
        c1c = (self.mnMatchesInliers < nRefMatches * 0.25 or bNeedToInsertClose)
        c2 = (self.mnMatchesInliers < nRefMatches * thRefRatio or bNeedToInsertClose) and self.mnMatchesInliers > 15

        if (c1a or c1b or c1c) and c2:
            # Insert keyframe if mapping accepts keyframes
            if bLocalMappingIdle:
                return True
            else:
                self.mpLocalMapper.interrupt_BA();

                if self.mpLocalMapper.keyframes_in_queue() < 3:
                    return True
                else:
                    return False
        else:
            return False


    def create_new_key_frame(self):
        """
        Create and insert a new keyframe into the map.
        """
        if not self.mpLocalMapper.set_not_stop(True):
            return

        # Create a new keyframe
        pKF = KeyFrame(self.mCurrentFrame, self.mpMap, self.mpKeyFrameDB)

        self.mpReferenceKF = pKF
        self.mCurrentFrame.mpReferenceKF = pKF

        self.mCurrentFrame.update_pose_matrices()

       # Sort points by measured depth from stereo/RGBD sensor
        vDepthIdx = []
        for i in range(self.mCurrentFrame.N):
            z = self.mCurrentFrame.mvDepth[i]
            if z > 0:
                vDepthIdx.append((z, i))

        if vDepthIdx:
            vDepthIdx.sort()  # Sort by depth

            nPoints = 0
            for depth, i in vDepthIdx:
                bCreateNew = False
                pMP = self.mCurrentFrame.mvpMapPoints[i]
                if not pMP:
                    bCreateNew = True

                elif pMP.observations() < 1:
                    bCreateNew = True
                    self.mCurrentFrame.mvpMapPoints[i] = None

                if bCreateNew:
                    x3D = self.mCurrentFrame.unproject_stereo(i)
                    pNewMP = MapPoint(x3D, pKF, self.mpMap, idxF=i, kframe_bool=True)
                    pNewMP.add_observation(pKF, i)
                    pKF.add_map_point(pNewMP, i)
                    pNewMP.compute_distinctive_descriptors()
                    pNewMP.update_normal_and_depth()
                    self.mpMap.add_map_point(pNewMP)
                    self.mCurrentFrame.mvpMapPoints[i] = pNewMP

                    nPoints += 1
                else:
                    nPoints += 1

                if depth > self.mThDepth and nPoints > 100:
                    break


        self.mpLocalMapper.insert_key_frame(pKF)
        self.mpLocalMapper.set_not_stop(False)

        mnLastKeyFrameId = self.mCurrentFrame.mnId
        mpLastKeyFrame = pKF

    def track_with_motion_model(self):
        """
        Track the current frame using the motion model.

        Returns:
            bool: True if tracking was successful, False otherwise.
        """
        matcher = ORBMatcher(0.9, True)

        # Update the last frame pose according to its reference keyframe
        # Create "visual odometry" points if in Localization Mode
        #print("len mvpMapPoints", len(self.mLastFrame.mvpMapPoints))
        #print("len mvbOutlier", len(self.mLastFrame.mvbOutlier))

        self.update_last_frame()

        self.mCurrentFrame.set_pose(self.mVelocity @ self.mLastFrame.mTcw)

        self.mCurrentFrame.mvpMapPoints = [None] * self.mCurrentFrame.N

        #j = 0
        #for i, o in self.mLastFrame.mvbOutlier.items():
        #    if o:
        #        j = j + 1

        #print("number of outliers", j)

        #print("len mvpMapPoints", len(self.mLastFrame.mvpMapPoints))
        #print("len mvbOutlier", len(self.mLastFrame.mvbOutlier))

        th = 7
        nmatches = matcher.search_by_projection_f_f(self.mCurrentFrame, self.mLastFrame, th)

        # If few matches, use a wider search window
        if nmatches < 20:
            self.mCurrentFrame.mvpMapPoints = [None] * self.mCurrentFrame.N
            nmatches = matcher.search_by_projection_f_f(self.mCurrentFrame, self.mLastFrame, 2 * th)

        if nmatches < 20:
            return False

        # Optimize the frame pose with all matches
        self.optimizer.pose_optimization(self.mCurrentFrame)

        # Discard outliers
        nmatches_map = 0
        for i in range(self.mCurrentFrame.N):
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

        #print("nmatches_map = ", nmatches_map)

        if self.mbOnlyTracking:
            mbVO = nmatches_map < 10
            return nmatches > 20

        return nmatches_map >= 10

    def update_last_frame(self):
        """
        Update the last frame's pose and create "visual odometry" map points if needed.
        """

        # Update pose according to the reference keyframe
        pRef = self.mLastFrame.mpReferenceKF
        Tlr = self.mlRelativeFramePoses[-1]

        self.mLastFrame.set_pose(Tlr @ pRef.get_pose())

        if self.mnLastKeyFrameId == self.mLastFrame.mnId or not self.mbOnlyTracking:
            return

        # Create "visual odometry" MapPoints
        # Sort points by measured depth from the stereo/RGB-D sensor
        vDepthIdx = []
        for i in range(self.mLastFrame.N):
            z = self.mLastFrame.mvDepth[i]
            if z > 0:
                vDepthIdx.append((z, i))

        if not vDepthIdx:
            return

        vDepthIdx.sort()  # Sort by depth

        # Insert all close points (depth < mThDepth)
        # If fewer than 100 close points, insert the 100 closest ones
        nPoints = 0
        for depth, i in vDepthIdx:
            bCreateNew = False

            if i not in self.mLastFrame.mvpMapPoints:
                bCreateNew = True
            elif self.mLastFrame.mvpMapPoints[i].observations() < 1:
                bCreateNew = True

            if bCreateNew:
                x3D = self.mLastFrame.unproject_stereo(i)
                pNewMP = MapPoint(x3D, self.mLastFrame, self.mpMap, idxF=i, kframe_bool=False)

                self.mLastFrame.mvpMapPoints[i] = pNewMP
                self.mLastFrame.mvbOutlier[i] = False
                self.mlpTemporalPoints.append(pNewMP)
                nPoints += 1
            else:
                nPoints += 1

            if depth > self.mThDepth and nPoints > 100:
                break

    def relocalization(self):
        # Compute Bag of Words Vector
        self.mCurrentFrame.compute_BoW()
        print("------------------------------------------------------------  relocalization")
        # Relocalization is performed when tracking is lost
        # Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vpCandidateKFs = self.mpKeyFrameDB.detect_relocalization_candidates(self.mCurrentFrame)
        print("vpCandidateKFs", len(vpCandidateKFs))
        if not vpCandidateKFs:
            return False

        nKFs = len(vpCandidateKFs)

        # Perform ORB matching with each candidate
        # If enough matches are found, setup a PnP solver
        matcher = ORBMatcher(0.75, True)

        vpPnPsolvers = [None] * nKFs
        vvpMapPointMatches = [None] * nKFs
        vbDiscarded = [False] * nKFs

        nCandidates = 0

        for i in range(nKFs):
            pKF = vpCandidateKFs[i]
            if pKF.is_bad():
                vbDiscarded[i] = True
            else:
                nmatches, vvpMapPointMatches[i] = matcher.search_by_BoW_kf_f(pKF, self.mCurrentFrame)
                if nmatches < 15:
                    vbDiscarded[i] = True
                    continue
                else:
                    pSolver = PnPsolver(self.mCurrentFrame, vvpMapPointMatches[i])
                    pSolver.set_ransac_parameters(0.99, 10, 300, 4, 0.4, 5.991)
                    vpPnPsolvers[i] = pSolver
                    nCandidates += 1

        # Perform P4P RANSAC iterations
        # Until a camera pose supported by enough inliers is found
        bMatch = False
        matcher2 = ORBMatcher(0.9, True)

        while nCandidates > 0 and not bMatch:
            for i in range(nKFs):
                if vbDiscarded[i]:
                    continue

                # Perform 5 RANSAC iterations
                vbInliers = []
                nInliers = 0
                bNoMore = False

                pSolver = vpPnPsolvers[i]
                Tcw, bNoMore, vbInliers, nInlier = pSolver.iterate(5)
                # If RANSAC reaches max iterations, discard keyframe
                if bNoMore:
                    vbDiscarded[i] = True
                    nCandidates -= 1

                # If a camera pose is computed, optimize
                if Tcw is not None:
                    self.mCurrentFrame.mTcw = Tcw

                    sFound = OrderedSet()

                    np = len(vbInliers)

                    for j in range(np):
                        if vbInliers[j]:
                            self.mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j]
                            sFound.add(vvpMapPointMatches[i][j])
                        else:
                            self.mCurrentFrame.mvpMapPoints[j] = None
                    nGood = self.optimizer.pose_optimization(self.mCurrentFrame)
                    if nGood < 10:
                        continue

                    for io in range(self.mCurrentFrame.N):
                        if self.mCurrentFrame.mvbOutlier[io]:
                            self.mCurrentFrame.mvpMapPoints[io] = None

                    # If few inliers, search by projection in a coarse window and optimize again
                    if nGood < 50:
                        nadditional = matcher2.search_by_projection_f_kf_f(
                            self.mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100
                        )

                        if nadditional + nGood >= 50:
                            nGood = self.optimizer.pose_optimization(self.mCurrentFrame)

                            # If many inliers but still not enough, search by projection again in a narrower window
                            if 30 < nGood < 50:
                                sFound.clear()
                                for ip in range(self.mCurrentFrame.N):
                                    if self.mCurrentFrame.mvpMapPoints[ip]:
                                        sFound.add(self.mCurrentFrame.mvpMapPoints[ip])
                                nadditional = matcher2.search_by_projection_f_kf_f(
                                    self.mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64
                                )

                                # Final optimization
                                if nGood + nadditional >= 50:
                                    nGood = self.optimizer.pose_optimization(self.mCurrentFrame)

                                    for io in range(self.mCurrentFrame.N):
                                        if self.mCurrentFrame.mvbOutlier[io]:
                                            self.mCurrentFrame.mvpMapPoints[io] = None

                    # If the pose is supported by enough inliers, stop RANSACs and continue
                    if nGood >= 50:
                        bMatch = True
                        break
        if not bMatch:
            return False
        else:
            self.mnLastRelocFrameId = self.mCurrentFrame.mnId
            return True


    def inform_only_tracking(self, flag):
        self.mbOnlyTracking = flag

    def reset(self):
        """
        Reset the entire tracking system, including mapping, loop closing, and database clearing.
        """
        print("System Resetting")
        if self.mpViewer:
            self.mpViewer.request_stop()
            while not self.mpViewer.is_stopped():
                time.sleep(0.003)  # Sleep for 3 milliseconds

        # Reset Local Mapping
        print("Resetting Local Mapper...")
        self.mpLocalMapper.request_reset()
        print("done")

        # Reset Loop Closing
        print("Resetting Loop Closing...")
        self.mpLoopClosing.request_reset()
        print("done")

        # Clear BoW Database
        print("Resetting Database...")
        self.mpKeyFrameDB.clear()
        print("done")

        # Clear Map (erases MapPoints and KeyFrames)
        self.mpMap.clear()

        # Reset global IDs
        self.KeyFrame.nNextId = 0
        self.Frame.nNextId = 0
        self.mState = "NO_IMAGES_YET"

        # Clear tracking-related lists
        self.mlRelativeFramePoses.clear()
        self.mlpReferences.clear()
        self.mlFrameTimes.clear()
        self.mlbLost.clear()

        if self.mpViewer:
            self.mpViewer.release()



if __name__ == "__main__":

    import yaml
    from pyDBoW.TemplatedVocabulary import TemplatedVocabulary

    sys.path.append("./pyORBExtractor/lib/")
    from pyORBExtractor import ORBextractor

    from stereo_kitti import LoadImages
    from KeyFrameDatabase import KeyFrameDatabase
    from Map import Map
    from FrameDrawer import FrameDrawer
    from MapDrawer import MapDrawer

    from System import System

    vocabulary = TemplatedVocabulary(k=5, L=3, weighting="TF_IDF", scoring="L1_NORM")
    vocabulary.load_from_text_file("./Vocabulary/ORBvoc.txt")

    with open("configs/KITTI00-02.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    leftImages, rightImages, timeStamps = LoadImages("00")
    nImages = len(leftImages)

    ss = {
           "pKF_ss": None, "pKF_ss_lock" : threading.Lock(),
           "mbStopped_ss": False, "mbStopped_ss_lock" : threading.Lock(),
           "mbStopRequested_ss": False, "mbStopRequested_ss_lock" : threading.Lock(),
           "mbAcceptKeyFrames_ss": True, "mbAcceptKeyFrames_ss_lock" : threading.Lock(),
           "mbAbortBA_ss": False, "mbAbortBA_ss_lock" : threading.Lock(),
           "len_mlNewKeyFrames_ss": None, "len_mlNewKeyFrames_ss_lock" : threading.Lock(),
           "mbNotStop_ss": False, "mbNotStop_ss_lock" : threading.Lock()
         }

    mpKeyFrameDatabase = KeyFrameDatabase(vocabulary)

    mpMap = Map()

    mpFrameDrawer = FrameDrawer(mpMap)
    mpMapDrawer = MapDrawer(mpMap, cfg)

    mpTracker = Tracking(System, vocabulary, mpFrameDrawer, mpMapDrawer,
                                  mpMap, mpKeyFrameDatabase, cfg, sensor="Stereo")

    for i in range(10):

        mleft = cv2.imread(leftImages[i], cv2.IMREAD_GRAYSCALE)
        mright = cv2.imread(rightImages[i], cv2.IMREAD_GRAYSCALE)
        timestamp = float(timeStamps[i])
        print("FFFFFFFFFFFFFFFFFFFrame = " , i)
        Twc = mpTracker.grab_image_stereo(mleft, mright, timestamp)
        #print(Twc)


