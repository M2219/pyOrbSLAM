import threading
import time
from ORBMatcher import ORBMatcher

class LocalMapping:

    mMutexReset = threading.Lock()

    def __init__(self, pMap):

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

        self.mpMap = pMap
        self.mlNewKeyFrames = []

        self.mlpRecentAddedMapPoints = []

    def keyframes_in_queue(self):
        with self.mMutexNewKFs:
            return len(self.mlNewKeyFrames)

    def run(self):
        """
        Main loop of the Local Mapping thread.
        """
        self.mbFinished = False

        while True:
            # Mark Local Mapping as busy
            self.set_accept_key_frames(False)
            time.sleep(0.1)
            print(self.mlNewKeyFrames)
            # Check if there are keyframes in the queue
            if self.check_new_key_frames():

                # Process new keyframe
                self.process_new_key_frame()

                # check recent MapPoints
                self.map_point_culling()

                # Triangulate new MapPoints
                self.create_new_map_points()
                print("before neighbors", self.check_new_key_frames())

                if not self.check_new_key_frames():
                    # Search in neighbor keyframes and fuse point duplications
                    print("search in neighbors")
                    self.search_in_neighbors()

                self.mbAbortBA = False

                print("after neighbors", self.check_new_key_frames())
                if not self.check_new_key_frames() and not self.stop_requested():
                    # Perform local bundle adjustment
                    print("keyframe in map", self.mpMap.key_frames_in_map())
                    if self.mpMap.key_frames_in_map() > 2:
                        print("optimizing")
                        Optimizer.local_bundle_adjustment(self.mpCurrentKeyFrame, self.mbAbortBA, self.mpMap)

                    # Cull redundant local keyframes
                    self.key_frame_culling()

                #self.mpLoopCloser.insert_key_frame(self.mpCurrentKeyFrame) not yet

            elif self.stop():
                # Safe area to stop
                while self.is_stopped() and not self.check_finish():
                    time.sleep(0.003)

                if self.check_finish():
                    break

            self.reset_if_requested()

            # Mark Local Mapping as available
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
            self.mpCurrentKeyFrame = self.mlNewKeyFrames.pop(0)

        self.mpCurrentKeyFrame.compute_BoW()

        vpMapPointMatches = self.mpCurrentKeyFrame.get_map_point_matches()

        for i, pMP in vpMapPointMatches.items():
            if not pMP.is_bad():
                if not pMP.is_in_key_frame(self.mpCurrentKeyFrame):
                    pMP.add_observation(self.mpCurrentKeyFrame, i)
                    pMP.update_normal_and_depth()
                    pMP.compute_distinctive_descriptors()
                else:
                    self.mlpRecentAddedMapPoints.append(pMP)

    def map_point_culling(self):
        """
        Perform culling of recently added MapPoints.

        Args:
            mlpRecentAddedMapPoints (list): List of recently added MapPoints.
            mpCurrentKeyFrame (KeyFrame): The current KeyFrame object.
            mbMonocular (bool): Flag indicating whether the system is monocular.
        """
        nCurrentKFid = self.mpCurrentKeyFrame.mnId

        # Threshold for observations
        nThObs = 3
        cnThObs = nThObs

        # Iterate over the list of recently added MapPoints
        lit = 0
        while lit < len(self.mlpRecentAddedMapPoints):
            pMP = self.mlpRecentAddedMapPoints[lit]

            if pMP.is_bad():
                # Remove MapPoint if it is bad
                self.mlpRecentAddedMapPoints.pop(lit)
            elif pMP.get_found_ratio() < 0.25:
                # Set MapPoint as bad if found ratio is less than 0.25
                pMP.set_bad_flag()
                self.mlpRecentAddedMapPoints.pop(lit)
            elif (nCurrentKFid - pMP.mnFirstKFid) >= 2 and pMP.observations() <= cnThObs:
                # Set MapPoint as bad if conditions on KF ID and observations are met
                pMP.set_bad_flag()
                self.mlpRecentAddedMapPoints.pop(lit)
            elif (nCurrentKFid - pMP.mnFirstKFid) >= 3:
                # Remove MapPoint if it has been around too long
                self.mlpRecentAddedMapPoints.pop(lit)
            else:
                lit += 1

    def create_new_map_points(self):
        nn = 10
        nnew = 0

        vpNeighKFs = self.mpCurrentKeyFrame.get_best_covisibility_key_frames(nn)

        matcher = ORBMatcher(0.6, False)

        # Retrieve current keyframe properties
        Rcw1 = self.mpCurrentKeyFrame.get_rotation()
        Rwc1 = Rcw1.T
        tcw1 = self.mpCurrentKeyFrame.get_translation()
        Ow1 = self.mpCurrentKeyFrame.get_camera_center()

        fx1, fy1, cx1, cy1 = self.mpCurrentKeyFrame.fx, self.mpCurrentKeyFrame.fy, self.mpCurrentKeyFrame.cx, self.mpCurrentKeyFrame.cy
        invfx1, invfy1 = self.mpCurrentKeyFrame.invfx, self.mpCurrentKeyFrame.invfy
        ratioFactor = 1.5 * self.mpCurrentKeyFrame.mfScaleFactor
        # Iterate through neighboring keyframes
        for pKF2 in vpNeighKFs:
            # Check for new keyframes
            if self.mpCurrentKeyFrame.check_new_key_frames():
                return

            Ow2 = pKF2.get_camera_center()
            vBaseline = Ow2 - Ow1
            baseline = np.linalg.norm(vBaseline)

            # Check baseline for monocular or stereo configurations
            if baseline < pKF2.mb:
                continue

            # Compute fundamental matrix
            F12 = self.compute_f12(self.mpCurrentKeyFrame, pKF2)

            # Find matches
            vMatchedIndices = matcher.search_for_triangulation(mpCurrentKeyFrame, pKF2, F12, epipolar_constraint=False)

            Rcw2 = pKF2.get_rotation()
            Rwc2 = Rcw2.T
            tcw2 = pKF2.get_translation()

            fx2, fy2, cx2, cy2 = pKF2.fx, pKF2.fy, pKF2.cx, pKF2.cy
            invfx2, invfy2 = pKF2.invfx, pKF2.invfy

            # Triangulate matches
            for idx1, idx2 in vMatchedIndices:
                kp1, kp1_ur = mpCurrentKeyFrame.mvKeysUn[idx1], mpCurrentKeyFrame.mvuRight[idx1]
                kp2, kp2_ur = pKF2.mvKeysUn[idx2], pKF2.mvuRight[idx2]

                xn1 = np.array([(kp1.pt[0] - cx1) * invfx1, (kp1.pt[1] - cy1) * invfy1, 1.0])
                xn2 = np.array([(kp2.pt[0] - cx2) * invfx2, (kp2.pt[1] - cy2) * invfy2, 1.0])

                ray1 = Rwc1 @ xn1
                ray2 = Rwc2 @ xn2
                cosParallaxRays = np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))

                # Handle stereo information
                cosParallaxStereo = min(
                    cosParallaxRays + 1,
                    np.cos(2 * np.arctan2(mpCurrentKeyFrame.mb / 2, mpCurrentKeyFrame.mvDepth[idx1])) if kp1_ur >= 0 else float('inf'),
                    np.cos(2 * np.arctan2(pKF2.mb / 2, pKF2.mvDepth[idx2])) if kp2_ur >= 0 else float('inf')
                )

                if cosParallaxRays < cosParallaxStereo and cosParallaxRays > 0 and (kp1_ur >= 0 or kp2_ur >= 0 or cosParallaxRays < 0.9998):
                    # Triangulate 3D point
                    A = np.vstack([
                        xn1[0] * Rcw1[2, :] - Rcw1[0, :],
                        xn1[1] * Rcw1[2, :] - Rcw1[1, :],
                        xn2[0] * Rcw2[2, :] - Rcw2[0, :],
                        xn2[1] * Rcw2[2, :] - Rcw2[1, :]
                    ])
                    _, _, vt = np.linalg.svd(A)
                    x3D = vt[-1, :3] / vt[-1, 3]

                    # Check depth and reprojection error
                    if not check_depth_and_reprojection(x3D, Rcw1, tcw1, fx1, fy1, cx1, cy1, kp1, kp1_ur, mpCurrentKeyFrame) or \
                       not check_depth_and_reprojection(x3D, Rcw2, tcw2, fx2, fy2, cx2, cy2, kp2, kp2_ur, pKF2):
                        continue

                    # Check scale consistency
                    normal1, normal2 = x3D - Ow1, x3D - Ow2
                    dist1, dist2 = np.linalg.norm(normal1), np.linalg.norm(normal2)

                    if dist1 == 0 or dist2 == 0:
                        continue

                    ratioDist = dist2 / dist1
                    ratioOctave = mpCurrentKeyFrame.mvScaleFactors[kp1.octave] / pKF2.mvScaleFactors[kp2.octave]

                    if ratioDist * ratioFactor < ratioOctave or ratioDist > ratioOctave * ratioFactor:
                        continue

                    # Create MapPoint
                    pMP = MapPoint(x3D, mpCurrentKeyFrame, mpMap)
                    pMP.add_observation(mpCurrentKeyFrame, idx1)
                    pMP.add_observation(pKF2, idx2)

                    mpCurrentKeyFrame.add_map_point(pMP, idx1)
                    pKF2.add_map_point(pMP, idx2)

                    pMP.compute_distinctive_descriptors()
                    pMP.update_normal_and_depth()

                    mpMap.add_map_point(pMP)
                    mlpRecentAddedMapPoints.append(pMP)
                    nnew += 1

    def compute_f12(self, pKF1, pKF2):
        # Get rotations and translations
        R1w = pKF1.get_rotation()
        t1w = pKF1.get_translation()
        R2w = pKF2.get_rotation()
        t2w = pKF2.get_translation()

        # Compute relative rotation and translation
        R12 = R1w @ R2w.T
        t12 = -R1w @ R2w.T @ t2w + t1w

        # Compute skew-symmetric matrix of t12
        t12x = self.skew_symmetric_matrix(t12)

        # Get intrinsic parameters
        K1 = pKF1.mK
        K2 = pKF2.mK

        # Compute the fundamental matrix
        F12 = np.linalg.inv(K1.T) @ t12x @ R12 @ np.linalg.inv(K2)

        return F12

    def skew_symmetric_matrix(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ], dtype=np.float)

    def search_in_neighbors(self):

        # Retrieve neighbor keyframes
        nn = 10
        vpNeighKFs = self.mpCurrentKeyFrame.get_best_covisibility_key_frames(nn)
        vpTargetKFs = []

        for pKFi in vpNeighKFs:
            if pKFi.is_bad() or pKFi.mnFuseTargetForKF == self.mpCurrentKeyFrame.mnId:
                continue
            vpTargetKFs.append(pKFi)
            pKFi.mnFuseTargetForKF = self.mpCurrentKeyFrame.mnId

            # Extend to some second neighbors
            vpSecondNeighKFs = pKFi.get_best_covisibility_key_frames(5)
            for pKFi2 in vpSecondNeighKFs:
                if (
                    pKFi2.is_bad()
                    or pKFi2.mnFuseTargetForKF == self.mpCurrentKeyFrame.mnId
                    or pKFi2.mnId == self.mpCurrentKeyFrame.mnId
                ):
                    continue
                vpTargetKFs.append(pKFi2)

        # Search matches by projection from current KF in target KFs
        vpMapPointMatches = self.mpCurrentKeyFrame.get_map_point_matches()
        matcher = ORBMatcher()
        for pKFi in vpTargetKFs:
            matcher.fuse_pkf_mp(pKFi, vpMapPointMatches, th=3.0)

        # Search matches by projection from target KFs in current KF
        vpFuseCandidates = []
        for pKFi in vpTargetKFs:
            vpMapPointsKFi = pKFi.get_map_point_matches()
            for i, pMP in vpMapPointsKFi.items():
                if pMP.is_bad() or pMP.mnFuseCandidateForKF == self.mpCurrentKeyFrame.mnId:
                    continue

                pMP.mnFuseCandidateForKF = self.mpCurrentKeyFrame.mnId
                vpFuseCandidates.append(pMP)

        matcher.fuse_pkf_mp(self.mpCurrentKeyFrame, vpFuseCandidates, th=3.0)
        # Update points
        vpMapPointMatches = self.mpCurrentKeyFrame.get_map_point_matches()
        for i, pMP in vpMapPointMatches.items():
            if not pMP.is_bad():
                pMP.compute_distinctive_descriptors()
                pMP.update_normal_and_depth()

        # Update connections in covisibility graph
        self.mpCurrentKeyFrame.update_connections()

    def key_frame_culling(self):
        """
        Perform keyframe culling to remove redundant keyframes.
        A keyframe is considered redundant if 90% of the MapPoints it sees
        are seen in at least 3 other keyframes in the same or finer scale.
        """
        # Get local keyframes
        vpLocalKeyFrames = self.mpCurrentKeyFrame.get_vector_covisible_key_frames()

        for pKF in vpLocalKeyFrames:
            if pKF.mnId == 0:
                continue

            vpMapPoints = pKF.get_map_point_matches()

            nObs = 3
            thObs = nObs
            nRedundantObservations = 0
            nMPs = 0

            for i, pMP in vpMapPoints.itesm():
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

            # Mark keyframe as bad if 90% of its MapPoints are redundant
            if nRedundantObservations > 0.9 * nMPs:
                pKF.set_bad_flag()

    def stop_requested(self):
        """
        Checks if a stop has been requested.

        Returns:
            bool: True if stop is requested, False otherwise.
        """
        with self.mMutexStop:
            return self.mbStopRequested

    def stop(self):
        """
        Attempts to stop the local mapping process.

        Returns:
            bool: True if the system stops, False otherwise.
        """
        with self.mMutexStop:
            if self.mbStopRequested and not self.mbNotStop:
                self.mbStopped = True
                print("Local Mapping STOP")
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

    def set_finish(self):
        with self.mMutexFinish:
            self.mbFinished = True

        with self.mMutexStop:
            self.mbStopped = True


if __name__ == "__main__":

    import yaml
    from pyDBoW.TemplatedVocabulary import TemplatedVocabulary
    from ORBExtractor import ORBExtractor
    from stereo_kitti import LoadImages
    from KeyFrameDatabase import KeyFrameDatabase
    from Map import Map
    from FrameDrawer import FrameDrawer
    from MapDrawer import MapDrawer
    from Tracking import Tracking
    import cv2
    vocabulary = TemplatedVocabulary(k=5, L=3, weighting="TF_IDF", scoring="L1_NORM")
    vocabulary.load_from_text_file("./Vocabulary/ORBvoc.txt")

    with open("configs/KITTI00-02.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    leftImages, rightImages, timeStamps = LoadImages("00")
    nImages = len(leftImages)

    mpKeyFrameDatabase = KeyFrameDatabase(vocabulary)

    mpMap = Map()

    mpFrameDrawer = FrameDrawer(mpMap)
    mpMapDrawer = MapDrawer(mpMap, cfg)

    mpLocalMapper = LocalMapping(mpMap, ss=ss)

    mpTracker = Tracking(False, vocabulary, mpFrameDrawer, mpMapDrawer,
                                  mpMap, mpKeyFrameDatabase, cfg, sensor="Stereo")

    mptLocalMapping_thread = threading.Thread(target=mpLocalMapper.run)
    # Start threads
    mptLocalMapping_thread.start()

    mbActivateLocalizationMode = False
    mbDeactivateLocalizationMode = False
    mbReset = False

    mMutexMode = threading.Lock()

    for i in range(10):

        mleft = cv2.imread(leftImages[i], cv2.IMREAD_GRAYSCALE)
        mright = cv2.imread(rightImages[i], cv2.IMREAD_GRAYSCALE)
        timestamp = float(timeStamps[i])

        with mMutexMode:
            if mbActivateLocalizationMode:
                mpLocalMapper.request_stop()

                # Wait until Local Mapping has effectively stopped
                while not mpLocalMapper.is_stopped():
                    time.sleep(0.001)

                mpTracker.inform_only_tracking(True)
                mbActivateLocalizationMode = False

            if mbDeactivateLocalizationMode:
                mpTracker.inform_only_tracking(False)
                mpLocalMapper.release()
                mbDeactivateLocalizationMode = False

        with LocalMapping.mMutexReset:  # Equivalent to after viewer and loop closing complete reset() function in the tracking
            if mbReset:
                mpTracker.reset()
                mbReset = False

        Twc = mpTracker.grab_image_stereo(mleft, mright, timestamp)
        print(Twc)

    mptLocalMapping_thread.join()
