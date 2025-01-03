import threading
import time

import numpy as np
import g2o

from ORBMatcher import ORBMatcher
from Converter import Converter
from ordered_set import OrderedSet
from Sim3Solver import Sim3Solver
from Optimizer import Optimizer

class LoopClosing:
    def __init__(self, pSys, pMap, pDB, pVoc):

        self.mMutexLoopQueue = threading.Lock()
        self.mMutexGBA = threading.Lock()
        self.mMutexReset = threading.Lock()
        self.mMutexFinish = threading.Lock()

        self.mlpLoopKeyFrameQueue = []
        self.mvConsistentGroups = []
        self.mvpEnoughConsistentCandidates = []
        self.mbResetRequested = False
        self.mbFinishRequested = False
        self.mbFinished = True
        self.mpMatchedKF = None
        self.mLastLoopKFid = 0
        self.mbRunningGBA = False
        self.mbFinishedGBA = True
        self.mbStopGBA = False
        self.mpThreadGBA = None
        self.mnFullBAIdx = 0
        self.mnCovisibilityConsistencyTh = 3
        self.mbFixScale = False

        self.mpMap = pMap
        self.mpKeyFrameDB = pDB
        self.mpORBVocabulary = pVoc
        self.mpSystem = pSys
        self.converter = Converter()
        self.optimizer = Optimizer()


    @property
    def mpLocalMapper(self):
        return self.mpSystem.mpLocalMapper

    @property
    def mpTracker(self):
        return self.mpSystem.mpTracker

    def run(self):
        self.mbFinished = False

        while True:
            time.sleep(0.2)
            # Check if there are keyframes in the queue
            if self.check_new_keyframes():
                # Detect loop candidates and check covisibility consistency
                if self.detect_loop():
                    # Compute similarity transformation [sR|t]
                    # In the stereo/RGBD case s=1
                    if self.compute_sim3():
                        # Perform loop fusion and pose graph optimization
                        self.correct_loop()

            self.reset_if_requested()

            if self.check_finish():
                break

            time.sleep(0.005)  # Sleep for 5 milliseconds

        self.set_finish()

    def insert_key_frame(self, pKF):
        with self.mMutexLoopQueue:  # Lock for thread safety
            if pKF.mnId != 0:
                self.mlpLoopKeyFrameQueue.append(pKF)

    def check_new_keyframes(self):
        """Check if there are new keyframes in the queue."""
        with self.mMutexLoopQueue:  # Thread-safe access to the queue
            return len(self.mlpLoopKeyFrameQueue) > 0

    def detect_loop(self):
        print("in detect")
        # Lock the queue and process the first keyframe
        with self.mMutexLoopQueue:
            self.mpCurrentKF = self.mlpLoopKeyFrameQueue.pop(0)
            self.mpCurrentKF.set_not_erase()

        # Check minimum keyframe conditions
        if self.mpCurrentKF.mnId < self.mLastLoopKFid + 10:  #for debug set 2:
            self.mpKeyFrameDB.add(self.mpCurrentKF)
            self.mpCurrentKF.set_erase()
            return False

        # Compute reference BoW similarity score
        vpConnectedKeyFrames = self.mpCurrentKF.get_vector_covisible_key_frames()
        currentBowVec = self.mpCurrentKF.mBowVec
        minScore = 1.0

        for pKF in vpConnectedKeyFrames:
            if pKF.is_bad():
                continue
            bowVec = pKF.mBowVec
            score = self.mpORBVocabulary.score(currentBowVec, bowVec)
            minScore = min(minScore, score)

        # Query database for loop candidates
        vpCandidateKFs = self.mpKeyFrameDB.detect_loop_candidates(self.mpCurrentKF, minScore)
        if not vpCandidateKFs:
            self.mpKeyFrameDB.add(self.mpCurrentKF)
            self.mvConsistentGroups.clear()
            self.mpCurrentKF.set_erase()
            return False

        # Check consistency of loop candidates
        self.mvpEnoughConsistentCandidates.clear()
        vCurrentConsistentGroups = []
        vbConsistentGroup = [False] * len(self.mvConsistentGroups)

        for pCandidateKF in vpCandidateKFs:
            spCandidateGroup = pCandidateKF.get_connected_key_frames()
            spCandidateGroup.add(pCandidateKF)

            bEnoughConsistent = False
            bConsistentForSomeGroup = False

            for iG, (sPreviousGroup, nPreviousConsistency) in enumerate(self.mvConsistentGroups):
                bConsistent = any(kf in sPreviousGroup for kf in spCandidateGroup)

                if bConsistent:
                    bConsistentForSomeGroup = True
                    nCurrentConsistency = nPreviousConsistency + 1

                    if not vbConsistentGroup[iG]:
                        vCurrentConsistentGroups.append((spCandidateGroup, nCurrentConsistency))
                        vbConsistentGroup[iG] = True

                    if nCurrentConsistency >= self.mnCovisibilityConsistencyTh and not bEnoughConsistent:
                        self.mvpEnoughConsistentCandidates.append(pCandidateKF)
                        bEnoughConsistent = True

            if not bConsistentForSomeGroup:
                vCurrentConsistentGroups.append((spCandidateGroup, 0))

        # Update consistent groups
        self.mvConsistentGroups = vCurrentConsistentGroups

        # Add current keyframe to database
        self.mpKeyFrameDB.add(self.mpCurrentKF)
        if not self.mvpEnoughConsistentCandidates:
            self.mpCurrentKF.set_erase()
            return False
        else:
            return True

    def compute_sim3(self):
        """Compute Sim3 transformation for loop closure."""
        nInitialCandidates = len(self.mvpEnoughConsistentCandidates)

        # Initialize data structures
        vpSim3Solvers = [None] * nInitialCandidates
        vvpMapPointMatches = []
        vbDiscarded = [False] * nInitialCandidates

        matcher = ORBMatcher(0.75, True)
        nCandidates = 0

        # Compute ORB matches for each candidate
        for i, pKF in enumerate(self.mvpEnoughConsistentCandidates):
            pKF.set_not_erase()

            if pKF.is_bad():
                vbDiscarded[i] = True
                continue

            nmatches, vvpMapPointMatches_ = matcher.search_by_BoW_kf_kf(self.mpCurrentKF, pKF)
            vvpMapPointMatches.append(vvpMapPointMatches_)

            if nmatches < 20:
                vbDiscarded[i] = True
                continue

            pSolver = Sim3Solver(self.mpCurrentKF, pKF, vvpMapPointMatches[i], self.mbFixScale)
            pSolver.set_ransac_parameters(0.99, 20, 300)
            vpSim3Solvers[i] = pSolver
            nCandidates += 1

        bMatch = False

        # Perform RANSAC iterations
        while nCandidates > 0 and not bMatch:
            for i, pKF in enumerate(self.mvpEnoughConsistentCandidates):
                if vbDiscarded[i]:
                    continue

                pSolver = vpSim3Solvers[i]
                vbInliers = []
                nInliers = 0
                bNoMore = False

                Scm = pSolver.iterate(5)

                if bNoMore:
                    vbDiscarded[i] = True
                    nCandidates -= 1

                if Scm is not None:
                    vpMapPointMatches = [None] * len(vvpMapPointMatches[i])
                    for j, inlier in enumerate(vbInliers):
                        if inlier:
                            vpMapPointMatches[j] = vvpMapPointMatches[i][j]

                    R = pSolver.get_estimated_rotation()
                    t = pSolver.get_estimated_translation()
                    s = pSolver.get_estimated_scale()

                    matches, vpMapPointMatches = matcher.search_by_sim3(self.mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5)

                    gScm = g2o.Sim3(R, t, s)
                    nInliers, gScm = self.optimizer.optimize_sim3(self.mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, self.mbFixScale)

                    if nInliers >= 20:
                        bMatch = True
                        self.mpMatchedKF = pKF
                        gSmw = g2o.Sim3(pKF.get_rotation(), pKF.get_translation(), 1.0)
                        self.mg2oScw = gScm * gSmw
                        self.mvpCurrentMatchedPoints = vpMapPointMatches
                        break

        if not bMatch:
            for pKF in self.mvpEnoughConsistentCandidates:
                pKF.set_erase()
            self.mpCurrentKF.set_erase()
            return False

        # Retrieve MapPoints from Loop KeyFrame and neighbors
        vpLoopConnectedKFs = self.mpMatchedKF.get_vector_covisible_key_frames()
        vpLoopConnectedKFs.append(self.mpMatchedKF)
        self.mvpLoopMapPoints = []

        for pKF in vpLoopConnectedKFs:
            vpMapPoints = pKF.get_map_point_matches()
            for pMP in vpMapPoints:
                if pMP and not pMP.is_bad() and pMP.mnLoopPointForKF != self.mpCurrentKF.mnId:
                    self.mvpLoopMapPoints.append(pMP)
                    pMP.mnLoopPointForKF = self.mpCurrentKF.mnId

        # Find more matches projecting with the computed Sim3
        self.mScws  = self.converter.sim3_to_mat(self.mg2oScw) #  check if suitbale for search
        matcher.search_by_projection_ckf_scw_mp(self.mpCurrentKF, self.mScws, self.mvpLoopMapPoints, self.mvpCurrentMatchedPoints, 10)

        # Check if enough matches exist to accept the loop
        nTotalMatches = sum(1 for pMP in self.mvpCurrentMatchedPoints if pMP)

        if nTotalMatches >= 40:
            for pKF in self.mvpEnoughConsistentCandidates:
                if pKF != self.mpMatchedKF:
                    pKF.set_erase()
            return True
        else:
            for pKF in self.mvpEnoughConsistentCandidates:
                pKF.set_erase()
            self.mpCurrentKF.set_erase()
            return False

    def correct_loop(self):
        print("Loop detected!")

        # Stop Local Mapping
        self.mpLocalMapper.request_stop()

        # Abort running Global Bundle Adjustment
        if self.is_running_gba():
            with self.mMutexGba:
                self.mbStopGBA = True

                if self.mpThreadGBA:
                    self.mpThreadGBA.join()
                    self.mpThreadGBA = None

        # Wait for Local Mapping to stop
        while not self.mpLocalMapper.is_stopped():
            time.sleep(0.001)  # 1 millisecond

        # Update connections for the current keyframe
        self.mpCurrentKF.update_connections()

        # Retrieve connected keyframes
        self.mvpCurrentConnectedKFs = self.mpCurrentKF.get_vector_covisible_key_frames()
        self.mvpCurrentConnectedKFs.append(self.mpCurrentKF)

        corrected_sim3 = {}
        non_corrected_sim3 = {}

        corrected_sim3[self.mpCurrentKF] = self.mg2oScw
        Twc = self.mpCurrentKF.get_pose_inverse()

        # Lock the map for updates
        with self.mpMap.mMutexMapUpdate:
            for pKFi in self.mvpCurrentConnectedKFs:
                Tiw = pKFi.get_pose()

                if pKFi != self.mpCurrentKF:
                    Tic = Tiw @ Twc
                    Ric = Tic[:3, :3]
                    tic = Tic[:3, 3]
                    g2oSic = g2o.Sim3(Ric, tic, 1)
                    g2oCorrectedSiw = g2oSic * self.mg2oScw
                    corrected_sim3[pKFi] = g2oCorrectedSiw

                Riw = Tiw[:3, :3]
                tiw = Tiw[:3, 3]
                g2oSiw = g2o.Sim3(Riw, tiw, 1)
                non_corrected_sim3[pKFi] = g2oSiw

            # Correct MapPoints
            for pKFi, g2oCorrectedSiw in corrected_sim3.items():
                g2oCorrectedSwi = g2oCorrectedSiw.inverse()
                g2oSiw = non_corrected_sim3[pKFi]

                vpMPsi = pKFi.get_map_point_matches()
                for pMPi in vpMPsi:
                    if pMPi and not pMPi.is_bad() and pMPi.mnCorrectedByKF != self.mpCurrentKF.mnId:
                        P3Dw = pMPi.get_world_pos()
                        correctedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(P3Dw))
                        pMPi.set_world_pos(np.expand_dims(correctedP3Dw, axis=1))
                        pMPi.mnCorrectedByKF = self.mpCurrentKF.mnId
                        pMPi.mnCorrectedReference = pKFi.mnId
                        pMPi.update_normal_and_depth()

                # Update keyframe pose with corrected Sim3
                eigR = g2oCorrectedSiw.rotation().matrix()
                eigt = g2oCorrectedSiw.translation()
                scale = g2oCorrectedSiw.scale()

                eigt /= scale
                correctedTiw = self.converter.RT_to_TF(eigR, eigt)

                pKFi.set_pose(correctedTiw)
                pKFi.update_connections()

            # Loop Fusion
            for i, pLoopMP in enumerate(self.mvpCurrentMatchedPoints):
                if pLoopMP:
                    pCurMP = self.mpCurrentKF.get_map_point(i)
                    if pCurMP:
                        pCurMP.replace(pLoopMP)
                    else:
                        self.mpCurrentKF.add_map_point(pLoopMP, i)
                        pLoopMP.add_observation(self.mpCurrentKF, i)
                        pLoopMP.compute_distinctive_descriptors()

        # Search and fuse MapPoints
        self.search_and_fuse(corrected_sim3)

        # Update connections in the covisibility graph
        loop_connections = {}
        for pKFi in self.mvpCurrentConnectedKFs:
            vpPreviousNeighbors = pKFi.get_vector_covisible_key_frames()
            pKFi.update_connections()
            loop_connections[pKFi] = pKFi.get_connected_key_frames() - OrderedSet(vpPreviousNeighbors) - OrderedSet(self.mvpCurrentConnectedKFs)

        # Optimize essential graph
        self.optimizer.optimize_essential_graph(self.mpMap, self.mpMatchedKF, self.mpCurrentKF, non_corrected_sim3, corrected_sim3, loop_connections, self.mbFixScale)

        self.mpMap.inform_new_big_change()

        # Add loop edge
        self.mpMatchedKF.add_loop_edge(self.mpCurrentKF)
        self.mpCurrentKF.add_loop_edge(self.mpMatchedKF)

        # Launch Global Bundle Adjustment
        self.mbRunningGBA = True
        self.mbFinishedGBA = False
        self.mbStopGBA = False
        self.mpThreadGBA = threading.Thread(target=self.run_global_bundle_adjustment, args=(self.mpCurrentKF.mnId,))
        self.mpThreadGBA.start()

        # Release Local Mapping
        self.mpLocalMapper.release()

        self.mLastLoopKFid = self.mpCurrentKF.mnId


    def search_and_fuse(self, corrected_poses_map):
        """
        Search and fuse loop closure map points into the current map.

        Parameters:
        - corrected_poses_map: A dictionary mapping KeyFrame to its corrected pose (g2o.Sim3).
        """
        matcher = ORBMatcher(0.8, True)

        for pKF, g2oScw in corrected_poses_map.items():
            # Convert g2o.Sim3 to OpenCV matrix
            cvScw = self.converter.sim3_to_mat(g2oScw)

            # Prepare replacement map points
            vpReplacePoints = [None] * len(self.mvpLoopMapPoints)

            # Fuse loop map points with keyframe
            matcher.fuse_kf_scw_mp(pKF, cvScw, self.mvpLoopMapPoints, 4, vpReplacePoints)

            # Acquire map mutex
            with self.mpMap.mMutexMapUpdate:
                nLP = len(self.mvpLoopMapPoints)
                for i in range(nLP):
                    pRep = vpReplacePoints[i]
                    if pRep:
                        pRep.replace(self.mvpLoopMapPoints[i])

    def run_global_bundle_adjustment(self, nLoopKF):
        print("Starting Global Bundle Adjustment")

        idx = self.mnFullBAIdx
        self.optimizer.global_bundle_adjustment(self.mpMap, 10, self.mbStopGBA, nLoopKF, False)

        # Update all MapPoints and KeyFrames
        # Local Mapping was active during BA, meaning there might be new keyframes
        # not included in the Global BA and not consistent with the updated map.
        # We need to propagate the correction through the spanning tree
        with self.mMutexGBA:
            if idx != self.mnFullBAIdx:
                return

            if not self.mbStopGBA:
                print("Global Bundle Adjustment finished")
                print("Updating map ...")
                self.mpLocalMapper.request_stop()

                # Wait until Local Mapping has effectively stopped
                while not self.mpLocalMapper.is_stopped() and not self.mpLocalMapper.is_finished():
                    time.sleep(0.001)

                # Get Map Mutex
                with self.mpMap.mMutexMapUpdate:
                    # Correct keyframes starting at map's first keyframe
                    lpKFtoCheck = list(self.mpMap.mvpKeyFrameOrigins)

                    while lpKFtoCheck:
                        pKF = lpKFtoCheck.pop(0)
                        sChilds = pKF.get_childs()
                        Twc = pKF.get_pose_inverse()

                        for pChild in sChilds:
                            if pChild.mnBAGlobalForKF != nLoopKF:
                                Tchildc = pChild.get_pose() @ Twc
                                pChild.mTcwGBA = Tchildc @ pKF.mTcwGBA
                                pChild.mnBAGlobalForKF = nLoopKF

                            lpKFtoCheck.append(pChild)

                        pKF.mTcwBefGBA = pKF.get_pose()
                        pKF.set_pose(pKF.mTcwGBA)

                    # Correct MapPoints
                    vpMPs = self.mpMap.get_all_map_points()

                    for pMP in vpMPs:
                        if pMP.is_bad():
                            continue

                        if pMP.mnBAGlobalForKF == nLoopKF:
                            # If optimized by Global BA, just update
                            pMP.set_world_pos(pMP.mPosGBA)
                        else:
                            # Update according to the correction of its reference keyframe
                            pRefKF = pMP.get_reference_key_frame()

                            if pRefKF.mnBAGlobalForKF != nLoopKF:
                                continue

                            # Map to non-corrected camera
                            Rcw = pRefKF.mTcwBefGBA[:3, :3]
                            tcw = pRefKF.mTcwBefGBA[:3, 3:4]
                            Xc = Rcw @ pMP.get_world_pos() + tcw

                            # Backproject using corrected camera
                            Twc = pRefKF.get_pose_inverse()
                            Rwc = Twc[:3, :3]
                            twc = Twc[:3, 3:4]

                            pMP.set_world_pos(Rwc @ Xc + twc)

                    self.mpMap.inform_new_big_change()
                    self.mpLocalMapper.release()

                    print("Map updated!")

            self.mbFinishedGBA = True
            self.mbRunningGBA = False

    def request_reset(self):
        """Request a reset and wait until it is processed."""
        with self.mMutexReset:  # Set the reset request flag
            self.mbResetRequested = True

        # Wait for the reset request to be processed
        while True:
            with self.mMutexReset:
                if not self.mbResetRequested:
                    break
            time.sleep(0.005)  # Sleep for 5 milliseconds

    def reset_if_requested(self):
        """Perform a reset if requested."""
        with self.mMutexReset:  # Check and handle reset request
            if self.mbResetRequested:
                self.mlpLoopKeyFrameQueue.clear()
                self.mLastLoopKFid = 0
                self.mbResetRequested = False

    def request_finish(self):
        """Set the finish request flag."""
        with self.mMutexFinish:  # Ensure thread safety
            self.mbFinishRequested = True

    def check_finish(self):
        """Check if a finish has been requested."""
        with self.mMutexFinish:  # Ensure thread safety
            return self.mbFinishRequested

    def set_finish(self):
        """Mark the process as finished."""
        with self.mMutexFinish:  # Ensure thread safety
            self.mbFinished = True

    def is_finished(self):
        """Check if the process has been marked as finished."""
        with self.mMutexFinish:  # Ensure thread safety
            return self.mbFinished

    def is_running_gba(self):
        """
        Checks if the Global Bundle Adjustment (GBA) is currently running.

        Returns:
            bool: True if GBA is running, False otherwise.
        """
        with self.mMutexGBA:
            return self.mbRunningGBA

    def is_finished_gba(self):
        """
        Checks if the Global Bundle Adjustment (GBA) has finished.

        Returns:
            bool: True if GBA has finished, False otherwise.
        """
        with self.mMutexGBA:
            return self.mbFinishedGBA



