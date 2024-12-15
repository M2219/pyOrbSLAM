import threading

class LocalMapping:
    def __init__(self, pMap, bMonocular, ss):

        self.ss = ss
        self.mMutexNewKFs = self.ss["pKF_ss_lock"]

        self.mMutexAccept = threading.Lock()
        self.mMutexReset = threading.Lock()
        self.mMutexFinish = threading.Lock()
        self.mMutexStop = threading.Lock()
        self.mMutexAccept = threading.Lock()

        self.mbMonocular = bMonocular
        self.mbResetRequested = False
        self.mbFinishRequested = False
        self.mbFinished = True
        self.mpMap = pMap

        self.mbAbortBA = False
        self.mbStopped = False
        self.mbStopRequested = False
        self.mbNotStop = False
        self.mbAcceptKeyFrames = True

        self.mlNewKeyFrames = []

        len_in_queue = keyframes_in_queue()
        print("len_in_queue", len_in_queue)

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
            self.insert_keyframe(self.ss["pKF_ss"])
            print(self.mlNewKeyFrames)

        """
            # Check if there are keyframes in the queue
            if self.check_new_key_frames():
                # Process new keyframe
                self.process_new_key_frame()

                # check recent MapPoints
                self.map_point_culling()

                # Triangulate new MapPoints
                self.create_new_map_points()

                if not self.check_new_key_frames():
                    # Search in neighbor keyframes and fuse point duplications
                    self.search_in_neighbors()

                self.mbAbortBA = False

                if not self.check_new_key_frames() and not self.stop_requested():
                    # Perform local bundle adjustment
                    if self.mpMap.key_frames_in_map() > 2:
                        Optimizer.local_bundle_adjustment(self.mpCurrentKeyFrame, self.mbAbortBA, self.mpMap)

                    # Cull redundant local keyframes
                    self.key_frame_culling()

                self.mpLoopCloser.insert_key_frame(self.mpCurrentKeyFrame)

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
        """

    def set_accept_key_frames(self, value):
        with self.mMutexAccept:
            self.mbAcceptKeyFrames = flag

    def check_new_key_frames(self):
        with self.mMutexNewKFs:
            return len(self.mlNewKeyFrames) > 0

    def process_new_key_frame(self):
        with self.mMutexNewKFs:
            self.mpCurrentKeyFrame = self.mlNewKeyFrames.pop(0)

        self.mpCurrentKeyFrame.compute_bow()

        vpMapPointMatches = self.mpCurrentKeyFrame.get_map_point_matches()

        for i, pMP in enumerate(vpMapPointMatches):
            if pMP and not pMP.is_bad():
                if not pMP.is_in_key_frame(self.mpCurrentKeyFrame):
                    pMP.add_observation(self.mpCurrentKeyFrame, i)
                    pMP.update_normal_and_depth()
                    pMP.compute_distinctive_descriptors()
                else:
                    self.mlpRecentAddedMapPoints.append(pMP)

    def map_point_culling(mlpRecentAddedMapPoints, mpCurrentKeyFrame, mbMonocular):
        """
        Perform culling of recently added MapPoints.

        Args:
            mlpRecentAddedMapPoints (list): List of recently added MapPoints.
            mpCurrentKeyFrame (KeyFrame): The current KeyFrame object.
            mbMonocular (bool): Flag indicating whether the system is monocular.
        """
        nCurrentKFid = mpCurrentKeyFrame.mnId

        # Threshold for observations
        nThObs = 2 if mbMonocular else 3
        cnThObs = nThObs

        # Iterate over the list of recently added MapPoints
        lit = 0
        while lit < len(mlpRecentAddedMapPoints):
            pMP = mlpRecentAddedMapPoints[lit]

            if pMP.is_bad():
                # Remove MapPoint if it is bad
                mlpRecentAddedMapPoints.pop(lit)
            elif pMP.get_found_ratio() < 0.25:
                # Set MapPoint as bad if found ratio is less than 0.25
                pMP.set_bad_flag()
                mlpRecentAddedMapPoints.pop(lit)
            elif (nCurrentKFid - pMP.mnFirstKFid) >= 2 and pMP.observations() <= cnThObs:
                # Set MapPoint as bad if conditions on KF ID and observations are met
                pMP.set_bad_flag()
                mlpRecentAddedMapPoints.pop(lit)
            elif (nCurrentKFid - pMP.mnFirstKFid) >= 3:
                # Remove MapPoint if it has been around too long
                mlpRecentAddedMapPoints.pop(lit)
            else:
                lit += 1

    def create_new_map_points(mpCurrentKeyFrame, vpNeighKFs, matcher, mbMonocular, ratioFactor, mlpRecentAddedMapPoints, mpMap):
        """
        Create new map points by triangulating matches between the current KeyFrame and its neighbors.

        Args:
            mpCurrentKeyFrame (KeyFrame): Current KeyFrame.
            vpNeighKFs (list[KeyFrame]): Neighbor KeyFrames.
            matcher (ORBmatcher): Feature matcher.
            mbMonocular (bool): Whether the system is monocular.
            ratioFactor (float): Ratio factor for scale consistency check.
            mlpRecentAddedMapPoints (list): Recently added MapPoints.
            mpMap (Map): Map object to store the new MapPoints.
        """
        nn = 20 if mbMonocular else 10
        nnew = 0

        # Retrieve current keyframe properties
        Rcw1 = mpCurrentKeyFrame.get_rotation()
        Rwc1 = Rcw1.T
        tcw1 = mpCurrentKeyFrame.get_translation()
        Ow1 = mpCurrentKeyFrame.get_camera_center()

        fx1, fy1, cx1, cy1 = mpCurrentKeyFrame.fx, mpCurrentKeyFrame.fy, mpCurrentKeyFrame.cx, mpCurrentKeyFrame.cy
        invfx1, invfy1 = mpCurrentKeyFrame.invfx, mpCurrentKeyFrame.invfy

        # Iterate through neighboring keyframes
        for pKF2 in vpNeighKFs:
            # Check for new keyframes
            if mpCurrentKeyFrame.check_new_keyframes():
                return

            Ow2 = pKF2.get_camera_center()
            vBaseline = Ow2 - Ow1
            baseline = np.linalg.norm(vBaseline)

            # Check baseline for monocular or stereo configurations
            if (not mbMonocular and baseline < pKF2.mb) or (mbMonocular and baseline / pKF2.compute_scene_median_depth(2) < 0.01):
                continue

            # Compute fundamental matrix
            F12 = compute_f12(mpCurrentKeyFrame, pKF2)

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

    def search_in_neighbors(mpCurrentKeyFrame, mbMonocular, matcher):
        """
        Search for matches in neighboring KeyFrames and fuse MapPoints.

        Args:
            mpCurrentKeyFrame (KeyFrame): The current KeyFrame.
            mbMonocular (bool): Whether the system is monocular.
            matcher (ORBmatcher): An instance of the ORB matcher.
        """
        # Retrieve neighbor keyframes
        nn = 20 if mbMonocular else 10
        vpNeighKFs = mpCurrentKeyFrame.get_best_covisibility_keyframes(nn)
        vpTargetKFs = []

        for pKFi in vpNeighKFs:
            if pKFi.is_bad() or pKFi.mnFuseTargetForKF == mpCurrentKeyFrame.mnId:
                continue
            vpTargetKFs.append(pKFi)
            pKFi.mnFuseTargetForKF = mpCurrentKeyFrame.mnId

            # Extend to some second neighbors
            vpSecondNeighKFs = pKFi.get_best_covisibility_keyframes(5)
            for pKFi2 in vpSecondNeighKFs:
                if (
                    pKFi2.is_bad()
                    or pKFi2.mnFuseTargetForKF == mpCurrentKeyFrame.mnId
                    or pKFi2.mnId == mpCurrentKeyFrame.mnId
                ):
                    continue
                vpTargetKFs.append(pKFi2)

        # Search matches by projection from current KF in target KFs
        vpMapPointMatches = mpCurrentKeyFrame.get_map_point_matches()
        for pKFi in vpTargetKFs:
            matcher.fuse(pKFi, vpMapPointMatches)

        # Search matches by projection from target KFs in current KF
        vpFuseCandidates = []
        for pKFi in vpTargetKFs:
            vpMapPointsKFi = pKFi.get_map_point_matches()
            for pMP in vpMapPointsKFi:
                if not pMP or pMP.is_bad() or pMP.mnFuseCandidateForKF == mpCurrentKeyFrame.mnId:
                    continue
                pMP.mnFuseCandidateForKF = mpCurrentKeyFrame.mnId
                vpFuseCandidates.append(pMP)

        matcher.fuse(mpCurrentKeyFrame, vpFuseCandidates)

        # Update points
        vpMapPointMatches = mpCurrentKeyFrame.get_map_point_matches()
        for pMP in vpMapPointMatches:
            if pMP and not pMP.is_bad():
                pMP.compute_distinctive_descriptors()
                pMP.update_normal_and_depth()

        # Update connections in covisibility graph
        mpCurrentKeyFrame.update_connections()

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

    def insert_keyframe(self, pKF):
        """
        Inserts a new KeyFrame into the list in a thread-safe manner.

        Args:
            pKF (KeyFrame): The KeyFrame to insert.
        """
        with self.mMutexNewKFs:
            self.mlNewKeyFrames.append(pKF)
            self.mbAbortBA = True
