import threading
import math

import numpy as np
import cv2

from ordered_set import OrderedSet
from bisect import bisect_right

class KeyFrame:
    nNextId = 0
    def __init__(self, F, pMap, pKFDB):
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

        self.mRelocScore = 0.0

        self.mpKeyFrameDB = pKFDB
        self.mpORBvocabulary = F.mpORBvocabulary
        self.mbFirstConnection = True
        self.mpParent = None
        self.mbNotErase = False
        self.mbToBeErased = False
        self.mbBad = False
        self.mHalfBaseline = F.mb / 2
        self.mpMap = pMap

        self.mspChildrens = []
        self.mspLoopEdges = []

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

        with self.mMutexFeatures:
            vpMP = self.mvpMapPoints.copy()

        for pMP in vpMP:

            if not pMP:
                continue

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

        if not KFcounter:
            return

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

        vPairs.sort(reverse=True, key=lambda x: x[0])

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
        with self.mMutexConnections:

            vPairs = [(weight, pKF) for pKF, weight in self.mConnectedKeyFrameWeights.items()]
            vPairs = sorted(vPairs, key=lambda x: x[0])
            self.mvpOrderedConnectedKeyFrames = [pKF for _, pKF in vPairs]
            self.mvOrderedWeights = [weight for weight, _ in vPairs]


    def get_weight(pKF):
        with self.mMutexConnections:
            if pKF in self.mConnectedKeyFrameWeights:
                return self.mConnectedKeyFrameWeights[pKF]
            else:
                return 0

    def get_connected_key_frames(self):
        with self.mMutexConnections:
            return OrderedSet(self.mConnectedKeyFrameWeights.keys())

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

            index = bisect_right(self.mvOrderedWeights, w, key=lambda x: -x)
            if index == len(self.mvOrderedWeights):
                return []
            else:
                return self.mvpOrderedConnectedKeyFrames[:index]

    def add_map_point(self, pMP, indx):
        with self.mMutexFeatures:
            self.mvpMapPoints[indx] = pMP

    def erase_map_point_match_by_index(self, idx):
        with self.mMutexFeatures:
           self.mvpMapPoints[idx] = None

    def erase_map_point_match(self, idx):
        if idx >=0:
            self.mvpMapPoints[idx] = None

    def erase_map_point_match_by_pmp(self, pMP):
        idx = pMP.get_index_in_key_frame(self)
        if idx >=0:
            self.mvpMapPoints[idx] = None


    def replace_map_point_match(self, idx, pMP):
        self.mvpMapPoints[idx] = pMP

    def get_map_points(self):
        with self.mMutexFeatures:
            s = []
            for pMP in self.mvpMapPoints:
                if not pMP:
                    continue

                if not pMP.is_bad():
                    s.append(pMP)

        return OrderedSet(s)

    def tracked_map_points(self, minObs):
        with self.mMutexFeatures:
            nPoints = 0
            for pMP in self.mvpMapPoints:
                if pMP:
                    if not pMP.is_bad():
                        if minObs > 0 and pMP.observations() >= minObs:
                            nPoints += 1
                        elif minObs == 0:
                            nPoints += 1
        return nPoints

    def get_map_point_matches(self):
        with self.mMutexFeatures:
            return self.mvpMapPoints.copy()

    def get_map_point(self, idx):
        with self.mMutexFeatures:
            return self.mvpMapPoints[idx]

    def add_child(self, pKF):
        with self.mMutexConnections:
            self.mspChildrens.append(pKF)

    def erase_child(self, pKF):
        with self.mMutexConnections:
            self.mspChildrens.remove(pKF)

    def change_parent(self, pKF):
        with self.mMutexConnections:
            self.mpParent = pKF
            pKF.add_child(self)

    def get_childs(self):
        with self.mMutexConnections:
            return OrderedSet(self.mspChildrens)

    def get_parent(self):
        with self.mMutexConnections:
            return self.mpParent

    def has_child(self, pKF):
        with self.mMutexConnections:
            return pKF in self.mspChildrens

    def add_loop_edge(self, pKF):
        with self.mMutexConnections:
            self.mbNotErase = True
            self.mspLoopEdges.append(pKF)

    def get_loop_edges(self):
        with self.mMutexConnections:
            return OrderedSet(self.mspLoopEdges)

    def set_not_erase(self):
        with self.mMutexConnections:
            self.mbNotErase = True

    def set_erase(self):
        with self.mMutexConnections:
            if not self.mspLoopEdges:
                self.mbNotErase = False

        if self.mbToBeErased:
            self.set_bad_flag()

    def set_bad_flag(self):
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

                self.mConnectedKeyFrameWeights.clear()
                self.mvpOrderedConnectedKeyFrames.clear()

                sParentCandidates = [self.mpParent]

                while self.mspChildrens:
                    bContinue = False
                    max_weight = -1
                    pC = None
                    pP = None

                    for pKF in self.mspChildrens.copy():
                        if pKF.is_bad():
                            continue

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
                        sParentCandidates.append(pC)
                        self.mspChildrens.remove(pC)
                    else:
                        break

                if self.mspChildrens:
                    for pKF in self.mspChildrens:
                        pKF.ChangeParent(self.mpParent)

                self.mpParent.erase_child(self)
                self.mTcp = self.Tcw @ self.mpParent.get_pose_inverse()
                self.mbBad = True

            self.mpMap.erase_key_frame(self)
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
            self.update_best_covisibles()

    def get_features_in_area(self, x, y, r):
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
        return self.mnMinX <= x < self.mnMaxX and self.mnMinY <= y < self.mnMaxY

    def unproject_stereo(self, i):
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

