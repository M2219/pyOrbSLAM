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

                if pKF.mvuRight[idx] < 0:  # Monocular observation
                    obs = np.array([kpUn.pt.x, kpUn.pt.y])
                    e = g2o.EdgeProjectXYZ2UV()
                    e.set_vertex(0, optimizer.vertex(id))
                    e.set_vertex(1, optimizer.vertex(pKF.mnId))
                    e.set_measurement(obs)
                    invSigma2 = pKF.mvInvLevelSigma2[kpUn.octave]
                    e.set_information(np.identity(2) * invSigma2)

                    if bRobust:
                        rk = g2o.RobustKernelHuber()
                        rk.set_delta(thHuber2D)
                        e.set_robust_kernel(rk)

                    e.fx = pKF.fx
                    e.fy = pKF.fy
                    e.cx = pKF.cx
                    e.cy = pKF.cy

                    optimizer.add_edge(e)
                else:  # Stereo observation
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

    def pose_optimization(self, pFrame, pInd):
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
                    print(obs)
                    e = g2o.EdgeStereoSE3ProjectXYZOnlyPose()
                    e.set_vertex(0, optimizer.vertex(0))
                    e.set_measurement(obs)
                    #invSigma2 = pFrame.mvInvLevelSigma2[pFrame.mvKeysUn[i].octave]
                    #e.set_information(np.identity(3) * invSigma2)
                    print("here")

                    #rk = g2o.RobustKernelHuber()
                    #e.set_robust_kernel(rk)
                    #rk.set_delta(deltaStereo)

                    #e.fx = pFrame.fx
                    #e.fy = pFrame.fy
                    #e.cx = pFrame.cx
                    #e.cy = pFrame.cy
                    #e.bf = pFrame.mbf
                    #e.Xw = pMP.GetWorldPos()
                    #optimizer.add_edge(e)
                    #vpEdgesStereo.append(e)
                    #vnIndexEdgeStereo.append(i)

        if nInitialCorrespondences < 3:
            return 0

        # Perform 4 rounds of optimization
        chi2Mono = [5.991] * 4
        chi2Stereo = [7.815] * 4
        its = [10] * 4

        nBad = 0
        for it in range(4):
            vSE3.set_estimate(pFrame.mTcw)
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
        pFrame.SetPose(SE3quat_recov)

        return nInitialCorrespondences - nBad
