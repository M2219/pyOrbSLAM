import threading
import math

import numpy as np
import cv2

from bisect import bisect_right

class KeyFrame:
    nNextId = 0  # Class-level variable for unique KeyFrame IDs

    def __init__(self, F, pMap, pKFDB):
        """
        Initializes a KeyFrame object.

        Args:
            F (Frame): The frame from which the KeyFrame is created.
            pMap (Map): The map to which this KeyFrame belongs.
            pKFDB (KeyFrameDatabase): The KeyFrame database.
        """
        self.mMutexPose = threading.Lock()
        self.mMutexConnections = threading.Lock()
        self.mMutexFeatures = threading.Lock()

        self.mnFrameId = F.mnId
        self.mTimeStamp = F.mTimeStamp
        self.mnGridCols = F.FRAME_GRID_COLS
        self.mnGridRows = F.FRAME_GRID_ROWS
        self.mfGridElementWidthInv = F.mfGridElementWidthInv
        self.mfGridElementHeightInv = F.mfGridElementHeightInv

        self.mnTrackReferenceForFrame = 0
        self.mnFuseTargetForKF = 0
        self.mnBALocalForKF = 0
        self.mnBAFixedForKF = 0
        self.mnLoopQuery = 0
        self.mnLoopWords = 0
        self.mspLoopEdges = set()
        self.mnRelocQuery = 0
        self.mnRelocWords = 0
        self.mnBAGlobalForKF = 0

        self.fx = F.fx
        self.fy = F.fy
        self.cx = F.cx
        self.cy = F.cy
        self.invfx = F.invfx
        self.invfy = F.invfy
        self.mbf = F.mbf
        self.mb = F.mb
        self.mThDepth = F.mThDepth
        self.N = F.N

        self.mvKeys = F.mvKeys
        self.mvKeysUn = F.mvKeysUn
        self.mvuRight = F.mvuRight
        self.mvDepth = F.mvDepth
        self.mDescriptors = F.mDescriptors.copy()
        self.mBowVec = F.mBowVec
        self.mFeatVec = F.mFeatVec

        self.mnScaleLevels = F.mnScaleLevels
        self.mfScaleFactor = F.mfScaleFactor
        self.mfLogScaleFactor = F.mfLogScaleFactor
        self.mvScaleFactors = F.mvScaleFactors
        self.mvLevelSigma2 = F.mvLevelSigma2
        self.mvInvLevelSigma2 = F.mvInvLevelSigma2

        self.mnMinX = F.mnMinX
        self.mnMinY = F.mnMinY
        self.mnMaxX = F.mnMaxX
        self.mnMaxY = F.mnMaxY

        self.mK = F.mK
        self.mvpMapPoints = F.mvpMapPoints
        self.mvbOutlier = F.mvbOutlier
        #print("self.mvpMapPoints", len(self.mvpMapPoints))
        #print("self.mvbOutlier", len(self.mvbOutlier))
        self.mpKeyFrameDB = pKFDB
        self.mpORBvocabulary = F.mpORBvocabulary
        self.mbFirstConnection = True
        self.mpParent = None
        self.mbNotErase = False
        self.mbToBeErased = False
        self.mbBad = False
        self.mHalfBaseline = F.mb / 2
        self.mpMap = pMap

        self.mspChildrens = set()
        self.mspLoopEdges = set()
        # Assign a unique ID
        self.mnId = KeyFrame.nNextId
        KeyFrame.nNextId += 1

        self.mGrid = F.mGrid

        if F.mTcw is not None:
            self.set_pose(F.mTcw)

        self.F = F

        self.mConnectedKeyFrameWeights = {}
        self.mvpOrderedConnectedKeyFrames = []
    def set_pose(self, Tcw_):
        with self.mMutexPose:
            self.Tcw = Tcw_.copy()

            Rcw = self.Tcw[:3, :3]
            tcw = self.Tcw[:3, 3].reshape(3, 1)

            Rwc = Rcw.T
            self.Ow = -np.dot(Rwc, tcw)
            self.Ow = self.Ow

            self.Twc = np.eye(4, dtype=self.Tcw.dtype)
            self.Twc[:3, :3] = Rwc
            self.Twc[:3, 3] = self.Ow.flatten()
            self.Twc = self.Twc

            center = np.array([[self.mHalfBaseline], [0], [0], [1]], dtype=np.float32)
            self.Cw = np.dot(self.Twc, center)
            self.Cw = self.Cw

    def compute_BoW(self):

        self.mBowVec, self.mFeatVec = self.mpORBvocabulary.transform(self.F.mDescriptors, 4)

    def get_pose(self):
        with self.mMutexPose:
            return self.Tcw.copy()

    def get_pose_inverse(self):
        with self.mMutexPose:
            return self.Twc.copy()

    def get_camera_center(self):
        with self.mMutexPose:
            return self.Ow.copy()

    def get_stereo_center(self):
        with self.mMutexPose:
            return self.Cw.copy()

    def get_rotation(self):
        with self.mMutexPose:
            return self.Tcw[:3, :3].copy()

    def get_translation(self):
        with self.mMutexPose:
            return self.Tcw[:3, 3].reshape(3, 1).copy()

    def update_connections(self):
        KFcounter = {}
        vpMP = []

        # Get a copy of the map points
        with self.mMutexFeatures:
            vpMP = self.mvpMapPoints.copy()

        # Count observations of MapPoints in other KeyFrames
        for i, pMP in vpMP.items():

            if pMP.is_bad():
                continue

            observations = pMP.get_observations()
            for pKF, _ in observations.items():
                if pKF.mnId == self.mnId:
                    continue

                if pKF in KFcounter:
                    KFcounter[pKF] += 1
                else:
                    KFcounter[pKF] = 1

        # Return if no connected KeyFrames
        if not KFcounter:
            return

        # Determine connections
        nmax = 0
        pKFmax = None
        threshold = 15
        vPairs = []

        for pKF, count in KFcounter.items():
            if count > nmax:
                nmax = count
                pKFmax = pKF
            if count >= threshold:
                vPairs.append((count, pKF))
                pKF.add_connection(self, count)

        if not vPairs:
            vPairs.append((nmax, pKFmax))
            pKFmax.add_connection(self, nmax)

        # Sort pairs by weight
        vPairs.sort(reverse=True, key=lambda x: x[0])

        # Update ordered connected KeyFrames and weights
        lKFs = [pair[1] for pair in vPairs]
        lWs = [pair[0] for pair in vPairs]

        with self.mMutexConnections:
            self.mConnectedKeyFrameWeights = KFcounter
            self.mvpOrderedConnectedKeyFrames = lKFs
            self.mvOrderedWeights = lWs

            if self.mbFirstConnection and self.mnId != 0:
                self.mpParent = self.mvpOrderedConnectedKeyFrames[0]
                self.mpParent.add_child(self)
                self.mbFirstConnection = False

    def add_connection(self, pKF, weight):

        with self.mMutexConnections:
            if pKF not in self.mConnectedKeyFrameWeights:
                self.mConnectedKeyFrameWeights[pKF] = weight
            elif self.mConnectedKeyFrameWeights[pKF] != weight:
                self.mConnectedKeyFrameWeights[pKF] = weight
            else:
                return

        self.update_best_covisibles()

    def update_best_covisibles(self):
        """
        Updates the lists of ordered connected KeyFrames and their corresponding weights.
        """
        with self.mMutexConnections:

            vPairs = [(weight, pKF) for pKF, weight in self.mConnectedKeyFrameWeights.items()]
            vPairs = sorted(vPairs, key=lambda x: x[0])
            self.mvpOrderedConnectedKeyFrames = [pKF for _, pKF in vPairs]
            self.mvOrderedWeights = [weight for weight, _ in vPairs]


    def get_weight(pKF):
        """
        Retrieves the weight of the connection to the given KeyFrame.

        Args:
            pKF (KeyFrame): The connected KeyFrame.

        Returns:
            int: The weight of the connection, or 0 if not connected.
        """
        with self.mMutexConnections:
            if pKF in self.mConnectedKeyFrameWeights:
                return self.mConnectedKeyFrameWeights[pKF]
            else:
                return 0

    def get_connected_key_frames(self):
        with self.mMutexConnections:
            return set(self.mConnectedKeyFrameWeights.keys())


    def get_vector_covisible_key_frames(self):
        with self.mMutexConnections:
            return self.mvpOrderedConnectedKeyFrames.copy()

    def get_best_covisibility_key_frames(self, N):
        if len(self.mvpOrderedConnectedKeyFrames) < N:
            return self.mvpOrderedConnectedKeyFrames.copy()
        else:
            return self.mvpOrderedConnectedKeyFrames[:N]

    def get_covisibles_by_weight(self, w):
        with self.mMutexConnections:
            if not self.mvpOrderedConnectedKeyFrames:
                return []

        n = bisect_right(self.mvOrderedWeights, w)
        if n == len(self.mvOrderedWeights):
            return []
        else:
            return self.mvpOrderedConnectedKeyFrames[:n]

    def add_map_point(self, pMP, indx):
        with self.mMutexFeatures:
            self.mvpMapPoints[indx] = pMP
            self.mvbOutlier[indx] = False

    def erase_map_point_match_by_index(self, idx):
        with self.mMutexFeatures:
            if idx in self.mvpMapPoints:
               del self.mvpMapPoints[idx]
               del self.mvbOutlier[idx]

    def erase_map_point_match(self, idx):
        if idx in self.mvpMapPoints:
            del self.mvpMapPoints[idx]
            del self.mvbOutlier[idx]


    def replace_map_point_match(self, idx, pMP):
        self.mvpMapPoints[idx] = pMP
        self.mvbOutlier[idx] = False

    def get_map_points(self):
        with self.mMutexFeatures:
            return {pMP for i, pMP in self.mvpMapPoints.items() if not pMP.is_bad()}

    def tracked_map_points(self, minObs):
        with self.mMutexFeatures:
            nPoints = 0
            for i, pMP in self.mvpMapPoints.items():
                if not pMP.is_bad():
                    if minObs > 0 and list(pMP.get_observations().values())[0] >= minObs:
                        nPoints += 1
                    elif minObs == 0:
                        nPoints += 1
            return nPoints

    def get_map_point_matches(self):
        with self.mMutexFeatures:
            return self.mvpMapPoints.copy()

    def get_map_point(self, idx):
        with self.mMutexFeatures:
            if idx in self.mvpMapPoints:
                return self.mvpMapPoints[idx]
            else:
                return None

    def add_child(self, pKF):
        with self.mMutexConnections:
            self.mspChildrens.add(pKF)

    def erase_child(self, pKF):
        with self.mMutexConnections:
            self.mspChildrens.discard(pKF)

    def change_parent(self, pKF):
        with self.mMutexConnections:
            self.mpParent = pKF
            pKF.AddChild(self)

    def get_childs(self):
        with self.mMutexConnections:
            return set(self.mspChildrens)

    def get_parent(self):
        with self.mMutexConnections:
            return self.mpParent

    def has_child(self, pKF):
        with self.mMutexConnections:
            return pKF in self.mspChildrens

    def add_loop_edge(self, pKF):
        with self.mMutexConnections:
            self.mbNotErase = True
            self.mspLoopEdges.add(pKF)

    def get_loop_edges(self):
        with self.mMutexConnections:
            return self.mspLoopEdges

    def set_not_erase(self):
        with self.mMutexConnections:
            self.mbNotErase = True

    def set_erase(self):
        with self.mMutexConnections:
            if not self.mspLoopEdges:
                self.mbNotErase = False

        if self.mbToBeErased:
            self.SetBadFlag()

    def set_bad_flag(self):
        """
        Marks the KeyFrame as bad, removes connections, updates the spanning tree, and erases references.
        """
        if self.mnId == 0:
            return
        elif self.mbNotErase:
            self.mbToBeErased = True
            return

        for pKF in list(self.mConnectedKeyFrameWeights.keys()):
            pKF.EraseConnection(self)

        for i, pMP in self.mvpMapPoints:
            pMP.EraseObservation(self)

        with self.mMutexConnections:
            with self.mMutexFeatures:
                # Clear connections and features
                self.mConnectedKeyFrameWeights.clear()
                self.mvpOrderedConnectedKeyFrames.clear()

                # Update Spanning Tree
                sParentCandidates = {self.mpParent}

                while self.mspChildrens:
                    bContinue = False
                    max_weight = -1
                    pC = None
                    pP = None

                    for pKF in self.mspChildrens.copy():
                        if pKF.is_bad():
                            continue

                        # Check if a parent candidate is connected to the keyframe
                        vpConnected = pKF.GetVectorCovisibleKeyFrames()
                        for spc in sParentCandidates:
                            for vp in vpConnected:
                                if vp.mnId == spc.mnId:
                                    weight = pKF.GetWeight(vp)
                                    if weight > max_weight:
                                        pC = pKF
                                        pP = vp
                                        max_weight = weight
                                        bContinue = True


                    if bContinue:
                        pC.ChangeParent(pP)
                        sParentCandidates.add(pC)
                        self.mspChildrens.remove(pC)
                    else:
                        break


                # Assign children with no covisibility links to the original parent
                if self.mspChildrens:
                    for pKF in self.mspChildrens:
                        pKF.ChangeParent(self.mpParent)

                # Remove from parent's children and mark as bad
                self.mpParent.EraseChild(self)
                self.mTcp = self.Tcw @ self.mpParent.GetPoseInverse()
                self.mbBad = True

            # Erase from the map and database
            self.mpMap.EraseKeyFrame(self)
            self.mpKeyFrameDB.erase(self)



    def is_bad(self):
        with self.mMutexConnections:
            return self.mbBad

    def erase_connection(self, pKF):
        bUpdate = False

        with self.mMutexConnections:

            if pKF in self.mConnectedKeyFrameWeights:
                del self.mConnectedKeyFrameWeights[pKF]
                bUpdate = True

        if bUpdate:
            self.UpdateBestCovisibles()

    def get_features_in_area(self, x, y, r):
        """
        Retrieves the indices of features within a specified area.

        Args:
            x (float): X-coordinate of the center.
            y (float): Y-coordinate of the center.
            r (float): Radius of the area.

        Returns:
            list: Indices of the features within the area.
        """
        vIndices = []
        nMinCellX = max(0, int(np.floor((x - self.mnMinX - r) * self.mfGridElementWidthInv)))
        if nMinCellX >= self.mnGridCols:
            return vIndices

        nMaxCellX = min(self.mnGridCols - 1, int(np.ceil((x - self.mnMinX + r) * self.mfGridElementWidthInv)))
        if nMaxCellX < 0:
            return vIndices

        nMinCellY = max(0, int(np.floor((y - self.mnMinY - r) * self.mfGridElementHeightInv)))
        if nMinCellY >= self.mnGridRows:
            return vIndices

        nMaxCellY = min(self.mnGridRows - 1, int(np.ceil((y - self.mnMinY + r) * self.mfGridElementHeightInv)))
        if nMaxCellY < 0:
            return vIndices

        for ix in range(nMinCellX, nMaxCellX + 1):
            for iy in range(nMinCellY, nMaxCellY + 1):
                vCell = self.mGrid[ix][iy]
                for idx in vCell:
                    kpUn = self.mvKeysUn[idx]
                    distx = kpUn.pt[0] - x
                    disty = kpUn.pt[1] - y
                    if abs(distx) < r and abs(disty) < r:
                        vIndices.append(idx)

        return vIndices

    def is_in_image(self, x, y):
        """
        Checks if a point is within the image bounds.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.

        Returns:
            bool: True if the point is within the image bounds, False otherwise.
        """
        return self.mnMinX <= x < self.mnMaxX and self.mnMinY <= y < self.mnMaxY

    def unproject_stereo(self, i):
        """
        Unprojects a 2D point to 3D using stereo depth.

        Args:
            i (int): Index of the point.

        Returns:
            np.ndarray: 3D coordinates in the world frame.
        """
        z = self.mvDepth[i]
        if z > 0:
            u = self.mvKeys[i].pt[0]
            v = self.mvKeys[i].pt[1]
            x = (u - self.cx) * z * self.invfx
            y = (v - self.cy) * z * self.invfy
            x3Dc = np.array([[x], [y], [z]])

            with self.mMutexPose:
                return self.Twc[:3, :3] @ x3Dc + self.Twc[:3, 3].reshape(3, 1)
        else:
            return None

    def compute_scene_median_depth(self, q):
        """
        Computes the median depth of the scene.

        Args:
            q (int): Quantile to compute the median depth.

        Returns:
            float: Median depth value.
        """
        with self.mMutexFeatures:
            with self.mMutexPose:
                vpMapPoints = self.mvpMapPoints.copy()
                Tcw_ = self.Tcw.copy()

        vDepths = []
        Rcw2 = Tcw_[:3, :3][2].reshape(1, 3)
        zcw = Tcw_[2, 3]
        for i in range(len(vpMapPoints)):
            pMP = vpMapPoints[i]
            if pMP:
                x3Dw = pMP.GetWorldPos()
                z = Rcw2 @ x3Dw + zcw
                vDepths.append(z)

        vDepths.sort()
        return vDepths[(len(vDepths) - 1) // q]

def compute_image_bounds(imLeft, mK, mDistCoef):

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



if __name__ == "__main__":

    import yaml
    from pyDBoW.TemplatedVocabulary import TemplatedVocabulary
    from ORBExtractor import ORBExtractor
    from stereo_kitti import LoadImages

    vocabulary = TemplatedVocabulary(k=5, L=3, weighting="TF_IDF", scoring="L1_NORM")
    vocabulary.load_from_text_file("./Vocabulary/ORBvoc.txt")

    with open("configs/KITTI00-02.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    leftImages, rightImages, timeStamps = LoadImages("00")
    nImages = len(leftImages)
    mleft = cv2.imread(leftImages[0], cv2.IMREAD_GRAYSCALE)
    mright = cv2.imread(rightImages[0], cv2.IMREAD_GRAYSCALE)
    timestamp = float(timeStamps[0])

    fx = cfg["Camera.fx"]
    fy = cfg["Camera.fy"]
    cx = cfg["Camera.cx"]
    cy = cfg["Camera.cy"]

    mk = np.eye(3, dtype=np.float32)
    mk[0][0] = fx
    mk[1][1] = fy
    mk[0][2] = cx
    mk[1][2] = cy


    mDistCoef = np.ones((1, 4), dtype=np.float32)
    mDistCoef[0][0] = cfg["Camera.k1"]
    mDistCoef[0][1] = cfg["Camera.k2"]
    mDistCoef[0][2] = cfg["Camera.p1"]
    mDistCoef[0][3] = cfg["Camera.p2"]

    mbf = cfg["Camera.bf"]
    mThDepth = mbf * cfg["ThDepth"] / fx;

    fps = cfg["Camera.fps"]
    if fps == 0:
        fps = 30

    mMinFrames = 0
    mMaxFrames = fps

    nRGB = cfg["Camera.RGB"]
    mbRGB = nRGB

    nFeatures = cfg["ORBextractor.nFeatures"]
    fScaleFactor = cfg["ORBextractor.scaleFactor"]
    nLevels = cfg["ORBextractor.nLevels"]
    fIniThFAST = cfg["ORBextractor.iniThFAST"]
    fMinThFAST = cfg["ORBextractor.minThFAST"]

    mORBExtractorLeft = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
    mORBExtractorRight = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)

    keypointsL = []
    keypointsL.append(cv2.KeyPoint(x=50.0, y=100.0, size=10.0, angle=45.0, response=0.8, octave=1, class_id=0))
    keypointsL.append(cv2.KeyPoint(x=150.0, y=200.0, size=12.5, angle=90.0, response=0.9, octave=2, class_id=1))
    keypointsL.append(cv2.KeyPoint(x=300.0, y=400.0, size=15.0, angle=-1.0, response=0.7, octave=3, class_id=2))


    keypointsR = []
    keypointsR.append(cv2.KeyPoint(x=54.0, y=140.0, size=13.0, angle=44.0, response=0.2, octave=1, class_id=0))
    keypointsR.append(cv2.KeyPoint(x=110.0, y=220.0, size=15.5, angle=20.0, response=0.4, octave=2, class_id=1))
    keypointsR.append(cv2.KeyPoint(x=200.0, y=440.0, size=17.0, angle=-11.0, response=0.6, octave=3, class_id=2))


    mnMinX, mnMaxX, mnMinY, mnMaxY = compute_image_bounds(mleft, mk, mDistCoef)

    FRAME_GRID_ROWS = 48
    FRAME_GRID_COLS = 64

    mfGridElementWidthInv = float(FRAME_GRID_COLS) / (mnMaxX - mnMinX)
    mfGridElementHeightInv = float(FRAME_GRID_ROWS) / (mnMaxY - mnMinY)

    invfx = 1.0 / fx
    invfy = 1.0 / fy

    # should be added to args
    frame_args = [fx, fy, cx, cy, invfx, invfy, mfGridElementWidthInv, mfGridElementHeightInv, mnMinX, mnMaxX, mnMinY, mnMaxY, FRAME_GRID_ROWS, FRAME_GRID_COLS]

    mTcw = np.eye(4, dtype=np.float32)
    """
    # Rotation
    mTcw[0, 0] = 0.866
    mTcw[0, 1] = -0.5
    mTcw[1, 0] = 0.5
    mTcw[1, 1] = 0.866
    mTcw[2, 2] = 1.0

    # Set translation
    mTcw[0, 3] = 0.5
    mTcw[1, 3] = 0.3
    mTcw[2, 3] = 1.0
    """
    mCurrentFrame = Frame(mleft, mright, timestamp, mORBExtractorLeft, mORBExtractorRight, vocabulary, mk, mDistCoef, mbf, mThDepth, mTcw, frame_args)

    mpMap = Map()
    pKFDB = None

    pKFini = KeyFrame(mCurrentFrame, mpMap, pKFDB)
    mpMap.add_key_frame(pKFini)
    print(mCurrentFrame.N)
    for i in range(mCurrentFrame.N):
        z = mCurrentFrame.mvDepth[i]
        if z > 0:
            # Unproject the 2D point to 3D
            x3D = mCurrentFrame.unproject_stereo(i)
            # Create a new MapPoint
            pNewMP = MapPoint(x3D, pKFini, mpMap)
            # Add observation to the new MapPoint
            pNewMP.add_observation(pKFini, i)
            # Associate the new MapPoint with the KeyFrame
            pKFini.add_map_point(pNewMP, i)
            # Compute descriptors and update normals and depth
            pNewMP.update_normal_and_depth()
            # Add the MapPoint to the map
            mpMap.add_map_point(pNewMP)
            # Associate the MapPoint with the current frame
            mCurrentFrame.mvpMapPoints[i] = pNewMP

    print(f"New map created with {mpMap.map_points_in_map()} points")
