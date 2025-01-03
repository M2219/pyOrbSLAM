import threading
import time
import numpy as np

from MapPoint import MapPoint
from ORBMatcher import ORBMatcher
from Optimizer import Optimizer

class LocalMapping:
    mMutexReset = threading.Lock()
    def __init__(self, pSys, pMap):

        self.mMutexNewKFs = threading.Lock()
        self.mMutexAccept = threading.Lock()
        self.mMutexFinish = threading.Lock()
        self.mMutexStop = threading.Lock()

        self.mbNotStop = False
        self.mbStopped = False
        self.mbAbortBA = False
        self.mbResetRequested = False
        self.mbFinishRequested = False
        self.mbFinished = True
        self.mbStopRequested = False
        self.mbAcceptKeyFrames = True

        self.mpSystem = pSys
        self.mpMap = pMap

        self.mlNewKeyFrames = []

        self.mlpRecentAddedMapPoints = []
        self.optimizer = Optimizer()

    @property
    def mpLoopCloser(self):
        return self.mpSystem.mpLoopCloser

    def keyframes_in_queue(self):
        with self.mMutexNewKFs:
            return len(self.mlNewKeyFrames)

    def run(self):
        self.mbFinished = False
        while True:

            self.set_accept_key_frames(False)
            time.sleep(0.1)
            if self.check_new_key_frames():

                self.process_new_key_frame()
                self.map_point_culling()
                self.create_new_map_points()

                if not self.check_new_key_frames():
                    self.search_in_neighbors()

                self.mbAbortBA = False

                if not self.check_new_key_frames() and not self.stop_requested():
                    if self.mpMap.key_frames_in_map() > 2:
                        self.optimizer.local_bundle_adjustment(self.mpCurrentKeyFrame, self.mbAbortBA, self.mpMap)

                    self.key_frame_culling()

                self.mpLoopCloser.insert_key_frame(self.mpCurrentKeyFrame)

            elif self.stop():
                while self.is_stopped() and not self.check_finish():
                    time.sleep(0.003)

                if self.check_finish():
                    break

            self.reset_if_requested()

            self.set_accept_key_frames(True)

            if self.check_finish():
                break

            time.sleep(0.003)

        self.set_finish()

    def insert_key_frame(self, pKF):
        with self.mMutexNewKFs:
            self.mlNewKeyFrames.append(pKF)
            self.mbAbortBA = True

    def set_accept_key_frames(self, flag):
        with self.mMutexAccept:
            self.mbAcceptKeyFrame = flag

    def accept_key_frames(self):
        with self.mMutexAccept:
            return self.mbAcceptKeyFrames

    def check_new_key_frames(self):
        with self.mMutexNewKFs:
            return len(self.mlNewKeyFrames) > 0

    def process_new_key_frame(self):
        with self.mMutexNewKFs:
            self.mpCurrentKeyFrame = self.mlNewKeyFrames[0]
            self.mlNewKeyFrames.pop(0)

        self.mpCurrentKeyFrame.compute_BoW()

        vpMapPointMatches = self.mpCurrentKeyFrame.get_map_point_matches()

        for i, pMP in enumerate(vpMapPointMatches):
            if pMP:
                if not pMP.is_bad():
                    if not pMP.is_in_key_frame(self.mpCurrentKeyFrame):
                        pMP.add_observation(self.mpCurrentKeyFrame, i)
                        pMP.update_normal_and_depth()
                        pMP.compute_distinctive_descriptors()
                    else:
                        self.mlpRecentAddedMapPoints.append(pMP)

        self.mpCurrentKeyFrame.update_connections();
        self.mpMap.add_key_frame(self.mpCurrentKeyFrame);

    def map_point_culling(self):
        nCurrentKFid = self.mpCurrentKeyFrame.mnId

        nThObs = 3
        cnThObs = nThObs

        lit = 0
        while lit < len(self.mlpRecentAddedMapPoints):
            pMP = self.mlpRecentAddedMapPoints[lit]

            if pMP.is_bad():
                self.mlpRecentAddedMapPoints.pop(lit)

            elif pMP.get_found_ratio() < 0.25:
                pMP.set_bad_flag()
                self.mlpRecentAddedMapPoints.pop(lit)

            elif (nCurrentKFid - pMP.mnFirstKFid) >= 2 and pMP.observations() <= cnThObs:
                pMP.set_bad_flag()
                self.mlpRecentAddedMapPoints.pop(lit)

            elif (nCurrentKFid - pMP.mnFirstKFid) >= 3:
                self.mlpRecentAddedMapPoints.pop(lit)

            else:
                lit += 1

    def create_new_map_points(self):
        nn = 10
        nnew = 0

        vpNeighKFs = self.mpCurrentKeyFrame.get_best_covisibility_key_frames(nn)

        matcher = ORBMatcher(0.6, False)

        Rcw1 = self.mpCurrentKeyFrame.get_rotation()
        Rwc1 = Rcw1.T
        tcw1 = self.mpCurrentKeyFrame.get_translation()
        Ow1 = self.mpCurrentKeyFrame.get_camera_center()
        Tcw1 = np.append(Rcw1, tcw1, 1)

        fx1, fy1, cx1, cy1 = self.mpCurrentKeyFrame.fx, self.mpCurrentKeyFrame.fy, self.mpCurrentKeyFrame.cx, self.mpCurrentKeyFrame.cy
        invfx1, invfy1 = self.mpCurrentKeyFrame.invfx, self.mpCurrentKeyFrame.invfy
        ratioFactor = 1.5 * self.mpCurrentKeyFrame.mfScaleFactor

        for i, pKF2 in enumerate(vpNeighKFs):
            if (i > 0) and (self.check_new_key_frames()):
                return

            Ow2 = pKF2.get_camera_center()
            vBaseline = Ow2 - Ow1
            baseline = np.linalg.norm(vBaseline)

            if baseline < pKF2.mb:
                continue

            F12 = self.compute_f12(self.mpCurrentKeyFrame, pKF2)

            vMatchedIndices = matcher.search_for_triangulation(self.mpCurrentKeyFrame, pKF2, F12, bOnlyStereo=False)

            Rcw2 = pKF2.get_rotation()
            Rwc2 = Rcw2.T
            tcw2 = pKF2.get_translation()

            Tcw2 = np.append(Rcw2, tcw2, 1)
            fx2, fy2, cx2, cy2 = pKF2.fx, pKF2.fy, pKF2.cx, pKF2.cy
            invfx2, invfy2 = pKF2.invfx, pKF2.invfy

            for idx1, idx2 in vMatchedIndices:
                kp1, kp1_ur = self.mpCurrentKeyFrame.mvKeysUn[idx1], self.mpCurrentKeyFrame.mvuRight[idx1]
                bStereo1 = kp1_ur>=0

                kp2, kp2_ur = pKF2.mvKeysUn[idx2], pKF2.mvuRight[idx2]
                bStereo2 = kp2_ur>=0

                xn1 = np.array([(kp1.pt[0] - cx1) * invfx1, (kp1.pt[1] - cy1) * invfy1, 1.0])
                xn2 = np.array([(kp2.pt[0] - cx2) * invfx2, (kp2.pt[1] - cy2) * invfy2, 1.0])

                ray1 = Rwc1 @ xn1
                ray2 = Rwc2 @ xn2
                cosParallaxRays = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))

                cosParallaxStereo = cosParallaxRays + 1
                cosParallaxStereo1 = cosParallaxStereo
                cosParallaxStereo2 = cosParallaxStereo

                if bStereo1:
                    cosParallaxStereo1 = np.cos(2 * np.arctan2(self.mpCurrentKeyFrame.mb / 2, self.mpCurrentKeyFrame.mvDepth[idx1]))
                elif bStereo2:
                    cosParallaxStereo2 = np.cos(2 * np.arctan2(pKF2.mb / 2, pKF2.mvDepth[idx2]))

                cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2)

                if cosParallaxRays < cosParallaxStereo and cosParallaxRays > 0 and (bStereo1 or bStereo2 or cosParallaxRays < 0.9998):
                    A = np.vstack([
                        xn1[0] * Tcw1[2, :] - Tcw1[0, :],
                        xn1[1] * Tcw1[2, :] - Tcw1[1, :],
                        xn2[0] * Tcw2[2, :] - Tcw2[0, :],
                        xn2[1] * Tcw2[2, :] - Tcw2[1, :]
                    ])

                    _, _, vt = np.linalg.svd(A)
                    x3D = np.expand_dims(vt[-1, 0:3] / vt[-1, 3], axis=0).T

                elif bStereo1 and cosParallaxStereo1 < cosParallaxStereo2:
                    x3D = self.mpCurrentKeyFrame.unproject_stereo(idx1)

                elif bStereo2 and cosParallaxStereo2 < cosParallaxStereo1:
                    x3D = pKF2.unproject_stereo(idx2)

                else:
                    continue

                z1 = np.dot(Rcw1[2:3, :], x3D)[0][0] + tcw1[2, 0]
                if z1 <= 0:
                    continue

                z2 = np.dot(Rcw2[2:3, :], x3D)[0][0] + tcw2[2, 0]
                if z2 <= 0:
                    continue

                sigmaSquare1 = self.mpCurrentKeyFrame.mvLevelSigma2[kp1.octave]
                x1 = np.dot(Rcw1[0:1, :], x3D)[0][0] + tcw1[0, 0]
                y1 = np.dot(Rcw1[1:2, :], x3D)[0][0] + tcw1[1, 0]
                invz1 = 1.0 / z1

                u1 = fx1 * x1 * invz1 + cx1
                v1 = fy1 * y1 * invz1 + cy1
                errX1 = u1 - kp1.pt[0]
                errY1 = v1 - kp1.pt[1]

                if not bStereo1:
                    if (errX1**2 + errY1**2) > 5.991 * sigmaSquare1:
                        continue
                else:
                    u1_r = u1 - self.mpCurrentKeyFrame.mbf * invz1
                    errX1_r = u1_r - kp1_ur
                    if (errX1**2 + errY1**2 + errX1_r**2) > 7.8 * sigmaSquare1:
                        continue

                sigmaSquare2 = pKF2.mvLevelSigma2[kp2.octave]
                x2 = np.dot(Rcw2[0:1, :], x3D)[0][0] + tcw2[0, 0]
                y2 = np.dot(Rcw2[1:2, :], x3D)[0][0] + tcw2[1, 0]
                invz2 = 1.0 / z2

                u2 = fx2 * x2 * invz2 + cx2
                v2 = fy2 * y2 * invz2 + cy2
                errX2 = u2 - kp2.pt[0]
                errY2 = v2 - kp2.pt[1]

                if not bStereo2:
                    if (errX2**2 + errY2**2) > 5.991 * sigmaSquare2:
                        continue
                else:
                    u2_r = u2 - self.mpCurrentKeyFrame.mbf * invz2
                    errX2_r = u2_r - kp2_ur
                    if (errX2**2 + errY2**2 + errX2_r**2) > 7.8 * sigmaSquare2:
                        continue

                normal1, normal2 = x3D - Ow1, x3D - Ow2
                dist1, dist2 = np.linalg.norm(normal1), np.linalg.norm(normal2)

                if dist1 == 0 or dist2 == 0:
                    continue

                ratioDist = dist2 / dist1
                ratioOctave = self.mpCurrentKeyFrame.mvScaleFactors[kp1.octave] / pKF2.mvScaleFactors[kp2.octave]

                if ratioDist * ratioFactor < ratioOctave or ratioDist > ratioOctave * ratioFactor:
                    continue

                pMP = MapPoint(x3D, self.mpCurrentKeyFrame, self.mpMap, idxF=idx1, kframe_bool=True)
                pMP.add_observation(self.mpCurrentKeyFrame, idx1)
                pMP.add_observation(pKF2, idx2)

                self.mpCurrentKeyFrame.add_map_point(pMP, idx1)
                pKF2.add_map_point(pMP, idx2)

                pMP.compute_distinctive_descriptors()
                pMP.update_normal_and_depth()

                self.mpMap.add_map_point(pMP)
                self.mlpRecentAddedMapPoints.append(pMP)
                nnew += 1

    def compute_f12(self, pKF1, pKF2):
        R1w = pKF1.get_rotation()
        t1w = pKF1.get_translation()
        R2w = pKF2.get_rotation()
        t2w = pKF2.get_translation()

        R12 = R1w @ R2w.T
        t12 = -R1w @ R2w.T @ t2w + t1w

        t12x = self.skew_symmetric_matrix(t12)

        K1 = pKF1.mK
        K2 = pKF2.mK

        F12 = np.linalg.inv(K1.T) @ t12x @ R12 @ np.linalg.inv(K2)

        return F12

    def skew_symmetric_matrix(self, v):
        return np.array([[0., -v[2][0], v[1][0]],
                         [v[2][0], 0., -v[0][0]],
                         [-v[1][0], v[0][0], 0.]])

    def search_in_neighbors(self):

        nn = 10
        vpNeighKFs = self.mpCurrentKeyFrame.get_best_covisibility_key_frames(nn)

        vpTargetKFs = []

        for pKFi in vpNeighKFs:
            if pKFi.is_bad() or pKFi.mnFuseTargetForKF == self.mpCurrentKeyFrame.mnId:
                continue

            vpTargetKFs.append(pKFi)
            pKFi.mnFuseTargetForKF = self.mpCurrentKeyFrame.mnId

            vpSecondNeighKFs = pKFi.get_best_covisibility_key_frames(5)
            for pKFi2 in vpSecondNeighKFs:
                if (
                    pKFi2.is_bad()
                    or pKFi2.mnFuseTargetForKF == self.mpCurrentKeyFrame.mnId
                    or pKFi2.mnId == self.mpCurrentKeyFrame.mnId
                ):
                    continue
                vpTargetKFs.append(pKFi2)

        vpMapPointMatches = self.mpCurrentKeyFrame.get_map_point_matches()
        matcher = ORBMatcher()
        for pKFi in vpTargetKFs:
            matcher.fuse_pkf_mp(pKFi, vpMapPointMatches, th=3.0)

        vpFuseCandidates = []
        for pKFi in vpTargetKFs:
            vpMapPointsKFi = pKFi.get_map_point_matches()
            for pMP in vpMapPointsKFi:
                if not pMP:
                    continue

                if pMP.is_bad() or pMP.mnFuseCandidateForKF == self.mpCurrentKeyFrame.mnId:
                    continue

                pMP.mnFuseCandidateForKF = self.mpCurrentKeyFrame.mnId
                vpFuseCandidates.append(pMP)

        matcher.fuse_pkf_mp(self.mpCurrentKeyFrame, vpFuseCandidates, th=3.0)
        vpMapPointMatches = self.mpCurrentKeyFrame.get_map_point_matches()
        for pMP in vpMapPointMatches:
            if pMP:
                if not pMP.is_bad():
                    pMP.compute_distinctive_descriptors()
                    pMP.update_normal_and_depth()

        self.mpCurrentKeyFrame.update_connections()

    def key_frame_culling(self):
        vpLocalKeyFrames = self.mpCurrentKeyFrame.get_vector_covisible_key_frames()

        for pKF in vpLocalKeyFrames:
            if pKF.mnId == 0:
                continue

            vpMapPoints = pKF.get_map_point_matches()

            nObs = 3
            thObs = nObs
            nRedundantObservations = 0
            nMPs = 0

            for i, pMP in enumerate(vpMapPoints):

                if pMP:

                    if not pMP.is_bad():
                        if pKF.mvDepth[i] > pKF.mThDepth or pKF.mvDepth[i] < 0:
                            continue

                        nMPs += 1
                        if pMP.observations() > thObs:
                            scaleLevel = pKF.mvKeysUn[i].octave
                            observations = pMP.get_observations()

                            nObs = 0
                            for pKFi, idx in observations.items():
                                if pKFi == pKF:
                                    continue
                                scaleLeveli = pKFi.mvKeysUn[idx].octave

                                if scaleLeveli <= scaleLevel + 1:
                                    nObs += 1
                                    if nObs >= thObs:
                                        break

                            if nObs >= thObs:
                                nRedundantObservations += 1

            if nRedundantObservations > 0.9 * nMPs:
                pKF.set_bad_flag()

    def request_reset(self):
        with self.mMutexReset:
            self.mbResetRequested = True

        while True:
            with self.mMutexReset:
                if not self.mbResetRequested:
                    break
            time.sleep(0.003)

    def request_stop(self):
        with self.mMutexStop:
            self.mbStopRequested = True

        with self.mMutexNewKFs:
            self.mbAbortBA = True

    def stop_requested(self):
        with self.mMutexStop:
            return self.mbStopRequested

    def release(self):
        with self.mMutexStop:
            with self.mMutexFinish:
                if self.mbFinished:
                    return
                self.mbStopped = False
                self.mbStopRequested = False
                for keyframe in self.mlNewKeyFrames:
                    del keyframe
                self.mlNewKeyFrames.clear()

    def stop(self):
        with self.mMutexStop:
            if self.mbStopRequested and not self.mbNotStop:
                self.mbStopped = True
                return True
            return False

    def set_not_stop(self, flag):
        with self.mMutexStop:
            if flag and self.mbStopped:
                return False

            self.mbNotStop = flag
            return True

    def is_stopped(self):
        with self.mMutexStop:
            return self.mbStopped

    def check_finish(self):
        with self.mMutexStop:
            return self.mbFinishRequested

    def reset_if_requested(self):
        with self.mMutexReset:
            if self.mbResetRequested:
                self.mlNewKeyFrames.clear()
                self.mlpRecentAddedMapPoints.clear()
                self.mbResetRequested = False

    def request_finish(self):
        with self.mMutexFinish:
            self.mbFinishRequested = True

    def set_finish(self):
        with self.mMutexFinish:
            self.mbFinished = True

        with self.mMutexStop:
            self.mbStopped = True

    def is_finished(self):
        with self.mMutexFinish:
            return self.mbFinished

