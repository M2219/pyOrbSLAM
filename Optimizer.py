import g2o
import numpy as np

from threading import Lock

from Convertor import Convertor
from MapPoint import MapPoint

class Optimizer:
    def __init__(self):

        self.mutex = Lock()
        self.convertor = Convertor()

    def global_bundle_adjustment(self, MapObj, nIterations, pbStopFlag=None, nLoopKF=0, bRobust=True):
        """
        Perform global bundle adjustment.

        Args:
            MapObj: The Map object containing all keyframes and map points.
            nIterations (int): Number of iterations for bundle adjustment.
            pbStopFlag (bool): Stop flag to terminate the optimization early (optional).
            nLoopKF (int): The loop keyframe identifier (default: 0).
            bRobust (bool): Whether to use robust kernel (default: True).
        """
        # Get all keyframes and map points from the map
        vpKFs = pMap.get_all_key_frames()
        vpMP = pMap.get_all_map_points()

        # Perform bundle adjustment
        BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust)


    def bundle_adjustment(self, vpKFs, vpMP, nIterations, pbStopFlag=None, nLoopKF=0, bRobust=True):
        """
        Perform bundle adjustment on the provided keyframes and map points.

        Args:
            vpKFs: List of KeyFrame objects.
            vpMP: List of MapPoint objects.
            nIterations: Number of iterations for the optimizer.
            pbStopFlag: Optional flag to stop optimization early.
            nLoopKF: KeyFrame ID for loop closure (default is 0).
            bRobust: Whether to use a robust kernel for optimization (default is True).
        """
        vbNotIncludedMP = [False] * len(vpMP)

        # Initialize the optimizer
        optimizer = g2o.SparseOptimizer()
        linear_solver = g2o.BlockSolverSE3.LinearSolverType()
        block_solver = g2o.BlockSolverSE3(linear_solver)
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)

        if pbStopFlag is not None:
            optimizer.set_force_stop_flag(pbStopFlag)

        maxKFid = 0

        # Set KeyFrame vertices
        for pKF in vpKFs:
            if pKF.is_bad():
                continue
            vSE3 = g2o.VertexSE3()
            vSE3.set_estimate(pKF.get_pose())
            vSE3.set_id(pKF.mnId)
            vSE3.set_fixed(pKF.mnId == 0)
            optimizer.add_vertex(vSE3)
            maxKFid = max(maxKFid, pKF.mnId)

        thHuber2D = np.sqrt(5.99)
        thHuber3D = np.sqrt(7.815)

        # Set MapPoint vertices
        for i, pMP in enumerate(vpMP):
            if pMP.is_bad():
                continue

            vPoint = g2o.VertexSBAPointXYZ()
            vPoint.set_estimate(pMP.get_world_pos())
            id = pMP.mnId + maxKFid + 1
            vPoint.set_id(id)
            vPoint.set_marginalized(True)
            optimizer.add_vertex(vPoint)

            observations = pMP.get_observations()
            nEdges = 0

            # Set edges
            for pKF, idx in observations.items():
                if pKF.is_bad() or pKF.mnId > maxKFid:
                    continue

                nEdges += 1
                kpUn = pKF.mvKeysUn[idx]

                obs = np.array([kpUn.pt.x, kpUn.pt.y, pKF.mvuRight[idx]])
                e = g2o.EdgeStereoProjectXYZ()
                e.set_vertex(0, optimizer.vertex(id))
                e.set_vertex(1, optimizer.vertex(pKF.mnId))
                e.set_measurement(obs)
                invSigma2 = pKF.mvInvLevelSigma2[kpUn.octave]
                e.set_information(np.identity(3) * invSigma2)

                if bRobust:
                    rk = g2o.RobustKernelHuber()
                    rk.set_delta(thHuber3D)
                    e.set_robust_kernel(rk)

                e.fx = pKF.fx
                e.fy = pKF.fy
                e.cx = pKF.cx
                e.cy = pKF.cy
                e.bf = pKF.mbf

                optimizer.add_edge(e)

            if nEdges == 0:
                optimizer.remove_vertex(vPoint)
                vbNotIncludedMP[i] = True
            else:
                vbNotIncludedMP[i] = False

        # Optimize
        optimizer.initialize_optimization()
        optimizer.optimize(nIterations)

        # Recover optimized data
        # KeyFrames
        for pKF in vpKFs:
            if pKF.is_bad():
                continue
            vSE3 = optimizer.vertex(pKF.mnId)
            SE3quat = vSE3.estimate()
            if nLoopKF == 0:
                pKF.set_pose(SE3quat)
            else:
                pKF.mTcwGBA = SE3quat
                pKF.mnBAGlobalForKF = nLoopKF

        # MapPoints
        for i, pMP in enumerate(vpMP):
            if vbNotIncludedMP[i]:
                continue

            if pMP.is_bad():
                continue
            vPoint = optimizer.vertex(pMP.mnId + maxKFid + 1)

            if nLoopKF == 0:
                pMP.set_world_pos(vPoint.estimate())
                pMP.update_normal_and_depth()
            else:
                pMP.mPosGBA = vPoint.estimate()
                pMP.mnBAGlobalForKF = nLoopKF

    def pose_optimization(self, pFrame):
        """
        Perform pose optimization for a single frame.

        Args:
            pFrame: The frame to optimize.

        Returns:
            int: The number of inliers after optimization.
        """
        # Initialize optimizer
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)

        nInitialCorrespondences = 0

        # Set Frame vertex
        vSE3 = g2o.VertexSE3Expmap()
        vSE3.set_estimate(self.convertor.to_se3_quat(pFrame.mTcw))
        vSE3.set_id(0)
        vSE3.set_fixed(False)
        optimizer.add_vertex(vSE3)

        # Prepare edge containers
        N = pFrame.N
        vpEdgesMono = []
        vnIndexEdgeMono = []
        vpEdgesStereo = []
        vnIndexEdgeStereo = []

        deltaMono = np.sqrt(5.991)
        deltaStereo = np.sqrt(7.815)

        # Add edges for map points
        with MapPoint.mGlobalMutex:  # Ensure thread safety
            for i, pMP in pFrame.mvpMapPoints.items():
                if pMP:
                    nInitialCorrespondences += 1
                    pFrame.mvbOutlier[i] = False

                    obs = np.array([pFrame.mvKeysUn[i].pt[0], pFrame.mvKeysUn[i].pt[1], pFrame.mvuRight[i]])
                    e = g2o.EdgeStereoSE3ProjectXYZOnlyPose()
                    e.set_vertex(0, optimizer.vertex(0))
                    e.set_measurement(obs)
                    invSigma2 = pFrame.mvInvLevelSigma2[pFrame.mvKeysUn[i].octave]
                    e.set_information(np.eye(3) * invSigma2)

                    e.set_robust_kernel(g2o.RobustKernelHuber(deltaStereo))

                    e.fx = pFrame.fx
                    e.fy = pFrame.fy
                    e.cx = pFrame.cx
                    e.cy = pFrame.cy
                    e.bf = pFrame.mbf
                    e.Xw = pMP.get_world_pos()
                    optimizer.add_edge(e)
                    vpEdgesStereo.append(e)
                    vnIndexEdgeStereo.append(i)

        if nInitialCorrespondences < 3:
            return 0

        # Perform 4 rounds of optimization
        chi2Mono = [5.991] * 4
        chi2Stereo = [7.815] * 4
        its = [10] * 4

        nBad = 0
        for it in range(4):
            vSE3.set_estimate(self.convertor.to_se3_quat(pFrame.mTcw))
            optimizer.initialize_optimization(0)
            optimizer.optimize(its[it])

            nBad = 0

            for e, idx in zip(vpEdgesMono, vnIndexEdgeMono):
                if pFrame.mvbOutlier[idx]:
                    e.compute_error()

                chi2 = e.chi2()
                if chi2 > chi2Mono[it]:
                    pFrame.mvbOutlier[idx] = True
                    e.set_level(1)
                    nBad += 1
                else:
                    pFrame.mvbOutlier[idx] = False
                    e.set_level(0)

                if it == 2:
                    e.set_robust_kernel(None)

            for e, idx in zip(vpEdgesStereo, vnIndexEdgeStereo):
                if pFrame.mvbOutlier[idx]:
                    e.compute_error()

                chi2 = e.chi2()
                if chi2 > chi2Stereo[it]:
                    pFrame.mvbOutlier[idx] = True
                    e.set_level(1)
                    nBad += 1
                else:
                    e.set_level(0)
                    pFrame.mvbOutlier[idx] = False

                if it == 2:
                    e.set_robust_kernel(None)

            if len(optimizer.edges()) < 10:
                break

        # Recover optimized pose
        vSE3_recov = optimizer.vertex(0)
        SE3quat_recov = vSE3_recov.estimate()
        pFrame.set_pose(self.convertor.to_mat(SE3quat_recov))

        return nInitialCorrespondences - nBad

    def local_bundle_adjustment(self, pKF, pbStopFlag, pMap):
        """
        Perform local bundle adjustment to optimize keyframes and map points.

        Args:
            pKF (KeyFrame): The current keyframe.
            pbStopFlag (bool): Flag to stop the optimization process.
            pMap (Map): The map containing the keyframes and map points.
        """
        # Local KeyFrames: First Breadth Search from Current Keyframe
        lLocalKeyFrames = [pKF]
        pKF.mnBALocalForKF = pKF.mnId

        vNeighKFs = pKF.get_vector_covisible_keyframes()
        for pKFi in vNeighKFs:
            pKFi.mnBALocalForKF = pKF.mnId
            if not pKFi.is_bad():
                lLocalKeyFrames.append(pKFi)

        # Local MapPoints seen in Local KeyFrames
        lLocalMapPoints = []
        for pKFi in lLocalKeyFrames:
            vpMPs = pKFi.get_map_point_matches()
            for pMP in vpMPs:
                if not pMP.is_bad() and pMP.mnBALocalForKF != pKF.mnId:
                    lLocalMapPoints.append(pMP)
                    pMP.mnBALocalForKF = pKF.mnId

        # Fixed Keyframes. Keyframes that see Local MapPoints but are not Local Keyframes
        lFixedCameras = []
        for pMP in lLocalMapPoints:
            observations = pMP.get_observations()
            for pKFi, _ in observations.items():
                if pKFi.mnBALocalForKF != pKF.mnId and pKFi.mnBAFixedForKF != pKF.mnId:
                    pKFi.mnBAFixedForKF = pKF.mnId
                    if not pKFi.is_bad():
                        lFixedCameras.append(pKFi)

        # Setup optimizer
        optimizer = g2o.SparseOptimizer()
        linear_solver = g2o.LinearSolverEigen(g2o.BlockSolverSE3.PoseMatrixType)
        solver_ptr = g2o.BlockSolverSE3(linear_solver)
        solver = g2o.OptimizationAlgorithmLevenberg(solver_ptr)
        optimizer.set_algorithm(solver)

        if pbStopFlag:
            optimizer.set_force_stop_flag(pbStopFlag)

        maxKFid = 0

        # Set Local KeyFrame vertices
        for pKFi in lLocalKeyFrames:
            vSE3 = g2o.VertexSE3Expmap()
            vSE3.set_estimate(self.convertor.to_se3_quat(pKFi.get_pose()))
            vSE3.set_id(pKFi.mnId)
            vSE3.set_fixed(pKFi.mnId == 0)
            optimizer.add_vertex(vSE3)
            maxKFid = max(maxKFid, pKFi.mnId)

        # Set Fixed KeyFrame vertices
        for pKFi in lFixedCameras:
            vSE3 = g2o.VertexSE3Expmap()
            vSE3.set_estimate(self.convertor.to_se3_quat(pKFi.get_pose()))
            vSE3.set_id(pKFi.mnId)
            vSE3.set_fixed(True)
            optimizer.add_vertex(vSE3)
            maxKFid = max(maxKFid, pKFi.mnId)

        # Set MapPoint vertices
        nExpectedSize = (len(lLocalKeyFrames) + len(lFixedCameras)) * len(lLocalMapPoints)
        vpEdgesMono = []
        vpEdgeKFMono = []
        vpMapPointEdgeMono = []

        vpEdgesStereo = []
        vpEdgeKFStereo = []
        vpMapPointEdgeStereo = []

        thHuberMono = np.sqrt(5.991)
        thHuberStereo = np.sqrt(7.815)

        for pMP in lLocalMapPoints:
            vPoint = g2o.VertexSBAPointXYZ()
            vPoint.set_estimate(to_vector3d(pMP.get_world_pos()))
            id = pMP.mnId + maxKFid + 1
            vPoint.set_id(id)
            vPoint.set_marginalized(True)
            optimizer.add_vertex(vPoint)

            observations = pMP.get_observations()
            for pKFi, idx in observations.items():
                if not pKFi.is_bad():
                    kpUn = pKFi.mvKeysUn[idx]

                    obs = np.array([kpUn.pt[0], kpUn.pt[1], pKFi.mvuRight[idx]])
                    e = g2o.EdgeStereoSE3ProjectXYZ()
                    e.set_vertex(0, optimizer.vertex(id))
                    e.set_vertex(1, optimizer.vertex(pKFi.mnId))
                    e.set_measurement(obs)
                    invSigma2 = pKFi.mvInvLevelSigma2[kpUn.octave]
                    e.set_information(np.identity(3) * invSigma2)
                    rk = g2o.RobustKernelHuber()
                    e.set_robust_kernel(rk)
                    rk.set_delta(thHuberStereo)
                    e.fx = pKFi.fx
                    e.fy = pKFi.fy
                    e.cx = pKFi.cx
                    e.cy = pKFi.cy
                    e.bf = pKFi.mbf
                    optimizer.add_edge(e)
                    vpEdgesStereo.append(e)
                    vpEdgeKFStereo.append(pKFi)
                    vpMapPointEdgeStereo.append(pMP)
                    obs = np.array([kpUn.pt[0], kpUn.pt[1], pKFi.mvuRight[idx]])
                    e = g2o.EdgeStereoSE3ProjectXYZ()
                    e.set_vertex(0, optimizer.vertex(id))
                    e.set_vertex(1, optimizer.vertex(pKFi.mnId))
                    e.set_measurement(obs)
                    invSigma2 = pKFi.mvInvLevelSigma2[kpUn.octave]
                    e.set_information(np.identity(3) * invSigma2)
                    rk = g2o.RobustKernelHuber()
                    e.set_robust_kernel(rk)
                    rk.set_delta(thHuberStereo)
                    e.fx = pKFi.fx
                    e.fy = pKFi.fy
                    e.cx = pKFi.cx
                    e.cy = pKFi.cy
                    e.bf = pKFi.mbf
                    optimizer.add_edge(e)
                    vpEdgesStereo.append(e)
                    vpEdgeKFStereo.append(pKFi)
                    vpMapPointEdgeStereo.append(pMP)

        if pbStopFlag and pbStopFlag[0]:
            return

        optimizer.initialize_optimization()
        optimizer.optimize(5)

        bDoMore = True

        if pbStopFlag and pbStopFlag[0]:
            bDoMore = False

        if bDoMore:
            # Check inlier observations for mono edges
            for e, pMP in zip(vpEdgesMono, vpMapPointEdgeMono):
                if pMP.is_bad():
                    continue
                if e.chi2() > 5.991 or not e.is_depth_positive():
                    e.set_level(1)
                e.set_robust_kernel(None)

            # Check inlier observations for stereo edges
            for e, pMP in zip(vpEdgesStereo, vpMapPointEdgeStereo):
                if pMP.is_bad():
                    continue
                if e.chi2() > 7.815 or not e.is_depth_positive():
                    e.set_level(1)
                e.set_robust_kernel(None)

            # Optimize again without the outliers
            optimizer.initialize_optimization(0)
            optimizer.optimize(10)

        vToErase = []

        # Check inlier observations and prepare points to erase
        for e, pMP, pKFi in zip(vpEdgesMono, vpMapPointEdgeMono, vpEdgeKFMono):
            if pMP.is_bad():
                continue
            if e.chi2() > 5.991 or not e.is_depth_positive():
                vToErase.append((pKFi, pMP))

        for e, pMP, pKFi in zip(vpEdgesStereo, vpMapPointEdgeStereo, vpEdgeKFStereo):
            if pMP.is_bad():
                continue
            if e.chi2() > 7.815 or not e.is_depth_positive():
                vToErase.append((pKFi, pMP))

        # Get Map Mutex
        with pMap.mMutexMapUpdate:  # Assuming this is a threading.Lock()
            for pKFi, pMPi in vToErase:
                pKFi.erase_map_point_match(pMPi)
                pMPi.erase_observation(pKFi)

        # Recover optimized data

        # Update KeyFrames
        for pKF in lLocalKeyFrames:
            vSE3 = optimizer.vertex(pKF.mnId)
            SE3quat = vSE3.estimate()
            pKF.set_pose(self.convertor.to_mat(SE3quat))

        # Update MapPoints
        for pMP in lLocalMapPoints:
            vPoint = optimizer.vertex(pMP.mnId + maxKFid + 1)
            pMP.set_world_pos(self.convertor.to_mat(vPoint.estimate()))
            pMP.update_normal_and_depth()
