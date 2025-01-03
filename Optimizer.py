import g2o
import numpy as np

from threading import Lock

from Converter import Converter
from MapPoint import MapPoint

class Optimizer:
    def __init__(self):

        self.mutex = Lock()
        self.converter = Converter()

    def global_bundle_adjustment(self, pMap, nIterations, pbStopFlag=None, nLoopKF=0, bRobust=True):
        vpKFs = pMap.get_all_key_frames()
        vpMP = pMap.get_all_map_points()
        self.bundle_adjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust)


    def bundle_adjustment(self, vpKFs, vpMP, nIterations, pbStopFlag=None, nLoopKF=0, bRobust=True):
        vbNotIncludedMP = [False] * len(vpMP)
        optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)

        if pbStopFlag is not None:
            optimizer.set_force_stop_flag(pbStopFlag)

        maxKFid = 0

        for pKF in vpKFs:
            if pKF.is_bad():
                continue
            vSE3 = g2o.VertexSE3Expmap()
            vSE3.set_estimate(self.converter.to_se3_quat(pKF.get_pose()))
            vSE3.set_id(pKF.mnId)
            vSE3.set_fixed(pKF.mnId == 0)
            optimizer.add_vertex(vSE3)
            maxKFid = max(maxKFid, pKF.mnId)

        thHuber2D = np.sqrt(5.99)
        thHuber3D = np.sqrt(7.815)

        for i, pMP in enumerate(vpMP):
            if pMP.is_bad():
                continue

            vPoint = g2o.VertexPointXYZ()
            vPoint.set_estimate(pMP.get_world_pos())
            id = pMP.mnId + maxKFid + 1
            vPoint.set_id(id)
            vPoint.set_marginalized(True)
            optimizer.add_vertex(vPoint)

            observations = pMP.get_observations()
            nEdges = 0

            for pKF, idx in observations.items():
                if pKF.is_bad() or pKF.mnId > maxKFid:
                    continue

                nEdges += 1
                kpUn = pKF.mvKeysUn[idx]

                obs = np.array([kpUn.pt[0], kpUn.pt[1], pKF.mvuRight[idx]])
                e = g2o.EdgeStereoSE3ProjectXYZ()
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

        optimizer.initialize_optimization()
        optimizer.optimize(nIterations)

        for pKF in vpKFs:
            if pKF.is_bad():
                continue
            vSE3 = optimizer.vertex(pKF.mnId)
            SE3quat = vSE3.estimate()
            if nLoopKF == 0:
                pKF.set_pose(self.converter.to_mat(SE3quat))
            else:
                pKF.mTcwGBA = self.converter.to_mat(SE3quat)
                pKF.mnBAGlobalForKF = nLoopKF

        for i, pMP in enumerate(vpMP):
            if vbNotIncludedMP[i]:
                continue

            if pMP.is_bad():
                continue
            vPoint = optimizer.vertex(pMP.mnId + maxKFid + 1)

            if nLoopKF == 0:
                pMP.set_world_pos(np.expand_dims(vPoint.estimate(), axis=1))
                pMP.update_normal_and_depth()
            else:
                pMP.mPosGBA = np.expand_dims(vPoint.estimate(), axis=1)
                pMP.mnBAGlobalForKF = nLoopKF

    def pose_optimization(self, pFrame):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(algorithm)

        nInitialCorrespondences = 0

        vSE3 = g2o.VertexSE3Expmap()
        vSE3.set_estimate(self.converter.to_se3_quat(pFrame.mTcw))
        vSE3.set_id(0)
        vSE3.set_fixed(False)
        optimizer.add_vertex(vSE3)

        N = pFrame.N
        vpEdgesStereo = []
        vnIndexEdgeStereo = []

        deltaStereo = np.sqrt(7.815)

        with MapPoint.mGlobalMutex:
            for i in range(pFrame.N):
                pMP = pFrame.mvpMapPoints[i]
                if pMP:
                    if pFrame.mvuRight[i] > 0:
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

        chi2Stereo = [7.815] * 4
        its = [10] * 4

        nBad = 0
        for it in range(4):
            vSE3.set_estimate(self.converter.to_se3_quat(pFrame.mTcw))
            optimizer.initialize_optimization(0)
            optimizer.optimize(its[it])

            nBad = 0

            for e, idx in zip(vpEdgesStereo, vnIndexEdgeStereo):
                if pFrame.mvbOutlier[idx]:
                    e.compute_error()

                chi2 = e.chi2()

                if (chi2 > chi2Stereo[it]) or (pFrame.mvuRight[idx] <= 0):
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

        vSE3_recov = optimizer.vertex(0)
        SE3quat_recov = vSE3_recov.estimate()
        pFrame.set_pose(self.converter.to_mat(SE3quat_recov))

        return nInitialCorrespondences - nBad

    def local_bundle_adjustment(self, pKF, pbStopFlag, pMap):
        lLocalKeyFrames = [pKF]
        pKF.mnBALocalForKF = pKF.mnId

        vNeighKFs = pKF.get_vector_covisible_key_frames()
        for pKFi in vNeighKFs:
            pKFi.mnBALocalForKF = pKF.mnId
            if not pKFi.is_bad():
                lLocalKeyFrames.append(pKFi)

        lLocalMapPoints = []
        for pKFi in lLocalKeyFrames:
            vpMPs = pKFi.get_map_point_matches()
            for pMP in vpMPs:
                if pMP:
                    if not pMP.is_bad() and pMP.mnBALocalForKF != pKF.mnId:
                        lLocalMapPoints.append(pMP)
                        pMP.mnBALocalForKF = pKF.mnId

        lFixedCameras = []
        for pMP in lLocalMapPoints:
            observations = pMP.get_observations()
            for pKFi, _ in observations.items():
                if pKFi.mnBALocalForKF != pKF.mnId and pKFi.mnBAFixedForKF != pKF.mnId:
                    pKFi.mnBAFixedForKF = pKF.mnId
                    if not pKFi.is_bad():
                        lFixedCameras.append(pKFi)

        optimizer = g2o.SparseOptimizer()
        solver_ptr = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver_ptr)
        optimizer.set_algorithm(solver)

        if pbStopFlag:
            optimizer.set_force_stop_flag(pbStopFlag)

        maxKFid = 0

        for pKFi in lLocalKeyFrames:
            vSE3 = g2o.VertexSE3Expmap()
            vSE3.set_estimate(self.converter.to_se3_quat(pKFi.get_pose()))
            vSE3.set_id(pKFi.mnId)
            vSE3.set_fixed(pKFi.mnId == 0)
            optimizer.add_vertex(vSE3)
            maxKFid = max(maxKFid, pKFi.mnId)

        for pKFi in lFixedCameras:
            vSE3 = g2o.VertexSE3Expmap()
            vSE3.set_estimate(self.converter.to_se3_quat(pKFi.get_pose()))
            vSE3.set_id(pKFi.mnId)
            vSE3.set_fixed(True)
            optimizer.add_vertex(vSE3)
            maxKFid = max(maxKFid, pKFi.mnId)

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

            vPoint = g2o.VertexPointXYZ()
            vPoint.set_estimate(pMP.get_world_pos())
            id = pMP.mnId + maxKFid + 1
            vPoint.set_id(id)
            vPoint.set_marginalized(True)
            optimizer.add_vertex(vPoint)

            observations = pMP.get_observations()
            for pKFi, idx in observations.items():
                if not pKFi.is_bad():
                    kpUn = pKFi.mvKeysUn[idx]
                    if pKFi.mvuRight[idx] > 0:
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
            for e, pMP in zip(vpEdgesMono, vpMapPointEdgeMono):
                if pMP.is_bad():
                    continue
                if e.chi2() > 5.991 or not e.is_depth_positive():
                    e.set_level(1)
                e.set_robust_kernel(None)

            for e, pMP in zip(vpEdgesStereo, vpMapPointEdgeStereo):
                if pMP.is_bad():
                    continue
                if e.chi2() > 7.815 or not e.is_depth_positive():
                    e.set_level(1)
                e.set_robust_kernel(None)

            optimizer.initialize_optimization(0)
            optimizer.optimize(10)

        vToErase = []

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

        with pMap.mMutexMapUpdate:
            for pKFi, pMPi in vToErase:
                pKFi.erase_map_point_match_by_pmp(pMPi)
                pMPi.erase_observation(pKFi)

        for pKF in lLocalKeyFrames:
            vSE3 = optimizer.vertex(pKF.mnId)
            SE3quat = vSE3.estimate()
            pKF.set_pose(self.converter.to_mat(SE3quat))

        for pMP in lLocalMapPoints:
            vPoint = optimizer.vertex(pMP.mnId + maxKFid + 1)
            pMP.set_world_pos(np.expand_dims(vPoint.estimate(), axis=1))
            pMP.update_normal_and_depth()

    def optimize_sim3(self, pKF1, pKF2, vpMatches1, g2oS12, th2, bFixScale):
        optimizer = g2o.SparseOptimizer()
        linear_solver = g2o.LinearSolverDenseX()
        solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverX(linear_solver))
        optimizer.set_algorithm(solver)

        K1 = pKF1.mK
        K2 = pKF2.mK

        R1w, t1w = pKF1.get_rotation(), pKF1.get_translation()
        R2w, t2w = pKF2.get_rotation(), pKF2.get_translation()

        vSim3 = g2o.VertexSim3Expmap()
        vSim3.set_id(0)
        vSim3.set_estimate(g2oS12)
        vSim3.set_fixed(False)

        fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
        fx2, fy2, cx2, cy2 = K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]

        vSim3._fix_scale = bFixScale
        vSim3._principle_point1 = np.array([cx1, cy1])
        vSim3._focal_length1 = np.array([fx1, fy1])
        vSim3._principle_point2 = np.array([cx2, cy2])
        vSim3._focal_length2 = np.array([fx2, fy2])

        optimizer.add_vertex(vSim3)

        vpEdges12 = []
        vpEdges21 = []
        vnIndexEdge = []

        delta_huber = np.sqrt(th2)
        nCorrespondences = 0

        for i, pMP1 in enumerate(pKF1.get_map_point_matches()):
            pMP2 = vpMatches1[i]

            if not pMP2 or not pMP1:
                continue

            id1, id2 = 2 * i + 1, 2 * (i + 1)

            i2 = pMP2.get_index_in_key_frame(pKF2)

            if pMP1.is_bad() or pMP2.is_bad() or i2 < 0:
                continue

            vPoint1 = g2o.VertexPointXYZ()
            vPoint1.set_id(id1)
            vPoint1.set_estimate(pMP1.get_world_pos())
            vPoint1.set_fixed(True)
            optimizer.add_vertex(vPoint1)

            vPoint2 = g2o.VertexPointXYZ()
            vPoint2.set_id(id2)
            vPoint2.set_estimate(pMP2.get_world_pos())
            vPoint2.set_fixed(True)
            optimizer.add_vertex(vPoint2)

            nCorrespondences += 1

            obs1 = np.array([pKF1.mvKeysUn[i].pt[0], pKF1.mvKeysUn[i].pt[1]])
            edge12 = g2o.EdgeSim3ProjectXYZ()
            edge12.set_vertex(0, optimizer.vertex(id2))
            edge12.set_vertex(1, optimizer.vertex(0))
            edge12.set_measurement(obs1)
            edge12.set_information(np.identity(2) * pKF1.mvInvLevelSigma2[pKF1.mvKeysUn[i].octave])
            edge12.set_robust_kernel(g2o.RobustKernelHuber())
            edge12.robust_kernel().set_delta(delta_huber)
            optimizer.add_edge(edge12)

            obs2 = np.array([pKF2.mvKeysUn[i2].pt[0], pKF2.mvKeysUn[i2].pt[1]])
            edge21 = g2o.EdgeInverseSim3ProjectXYZ()
            edge21.set_vertex(0, optimizer.vertex(id1))
            edge21.set_vertex(1, optimizer.vertex(0))
            edge21.set_measurement(obs2)
            edge21.set_information(np.identity(2) * pKF2.mvInvLevelSigma2[pKF2.mvKeysUn[i2].octave])
            edge21.set_robust_kernel(g2o.RobustKernelHuber())
            edge21.robust_kernel().set_delta(delta_huber)
            optimizer.add_edge(edge21)

            vpEdges12.append(edge12)
            vpEdges21.append(edge21)
            vnIndexEdge.append(i)

        optimizer.initialize_optimization()
        optimizer.optimize(5)

        nBad = 0
        for i, (edge12, edge21) in enumerate(zip(vpEdges12, vpEdges21)):
            if edge12.chi2() > th2 or edge21.chi2() > th2:
                idx = vnIndexEdge[i]
                vpMatches1[idx] = None
                optimizer.remove_edge(edge12)
                optimizer.remove_edge(edge21)
                nBad += 1

        if nCorrespondences - nBad < 10:
            return 0, g2oS12


        optimizer.initialize_optimization()
        optimizer.optimize(10 if nBad > 0 else 5)

        nInliers = 0
        for i, (edge12, edge21) in enumerate(zip(vpEdges12, vpEdges21)):
            if edge12.chi2() <= th2 and edge21.chi2() <= th2:
                nInliers += 1
            else:
                vpMatches1[vnIndexEdge[i]] = None

        vSim3_recov = optimizer.vertex(0)
        sim3 = vSim3_recov.estimate()

        return nInliers, sim3

    def optimize_essential_graph(self, pMap, pLoopKF, pCurKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, bFixScale):
        optimizer = g2o.SparseOptimizer()
        optimizer.set_verbose(False)

        linear_solver = g2o.LinearSolverEigenSim3()
        solver = g2o.BlockSolverSim3(linear_solver)
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        algorithm.set_user_lambda_init(1e-16)
        optimizer.set_algorithm(algorithm)

        vpKFs = pMap.get_all_key_frames()
        vpMPs = pMap.get_all_map_points()
        nMaxKFid = pMap.get_max_kf_id()

        vScw = [None] * (nMaxKFid + 1)
        vCorrectedSwc = [None] * (nMaxKFid + 1)
        vpVertices = [None] * (nMaxKFid + 1)

        minFeat = 100

        for pKF in vpKFs:
            if pKF.is_bad():
                continue

            vSim3 = g2o.VertexSim3Expmap()
            nIDi = pKF.mnId

            if pKF in CorrectedSim3:
                vScw[nIDi] = CorrectedSim3[pKF]
                vSim3.set_estimate(CorrectedSim3[pKF])
            else:
                Rcw = pKF.get_rotation()
                tcw = pKF.get_translation()
                Siw = g2o.Sim3(Rcw, tcw, 1.0)
                vScw[nIDi] = Siw
                vSim3.set_estimate(Siw)

            if pKF == pLoopKF:
                vSim3.set_fixed(True)

            vSim3.set_id(nIDi)
            vSim3.set_marginalized(False)
            vSim3._fix_scale = bFixScale

            optimizer.add_vertex(vSim3)
            vpVertices[nIDi] = vSim3

        sInsertedEdges = set()
        matLambda = np.identity(7)

        for pKF, spConnections in LoopConnections.items():
            nIDi = pKF.mnId
            Siw = vScw[nIDi]
            Swi = Siw.inverse()

            for pKF2 in spConnections:
                nIDj = pKF2.mnId

                if (nIDi != pCurKF.mnId or nIDj != pLoopKF.mnId) and pKF.get_weight(pKF2) < minFeat:
                    continue

                Sjw = vScw[nIDj]
                Sji = Sjw * Swi

                edge = g2o.EdgeSim3()
                edge.set_vertex(1, optimizer.vertex(nIDj))
                edge.set_vertex(0, optimizer.vertex(nIDi))
                edge.set_measurement(Sji)
                edge.set_information(matLambda)

                optimizer.add_edge(edge)
                sInsertedEdges.add((min(nIDi, nIDj), max(nIDi, nIDj)))

        for pKF in vpKFs:
            nIDi = pKF.mnId

            if pKF in NonCorrectedSim3:
                Swi = NonCorrectedSim3[pKF].inverse()
            else:
                Swi = vScw[nIDi].inverse()

            pParentKF = pKF.get_parent()

            if pParentKF:
                nIDj = pParentKF.mnId

                if pParentKF in NonCorrectedSim3:
                    Sjw = NonCorrectedSim3[pParentKF]
                else:
                    Sjw = vScw[nIDj]

                Sji = Sjw * Swi

                edge = g2o.EdgeSim3()
                edge.set_vertex(1, optimizer.vertex(nIDj))
                edge.set_vertex(0, optimizer.vertex(nIDi))
                edge.set_measurement(Sji)
                edge.set_information(matLambda)
                optimizer.add_edge(edge)

            for pLKF in pKF.get_loop_edges():
                if pLKF.mnId < pKF.mnId:
                    if pLKF in NonCorrectedSim3:
                        Slw = NonCorrectedSim3[pLKF]
                    else:
                        Slw = vScw[pLKF.mnId]

                    Sli = Slw * Swi
                    edge = g2o.EdgeSim3()
                    edge.set_vertex(1, optimizer.vertex(pLKF.mnId))
                    edge.set_vertex(0, optimizer.vertex(nIDi))
                    edge.set_measurement(Sli)
                    edge.set_information(matLambda)
                    optimizer.add_edge(edge)

            for pKFn in pKF.get_covisibles_by_weight(minFeat):
                if pKFn and pKFn != pParentKF and not pKF.has_child(pKFn) and not pKF.is_in_loop_edges(pKFn):
                    if not pKFn.is_bad() and pKFn.mnId < pKF.mnId:
                        if (min(pKF.mnId, pKFn.mnId), max(pKF.mnId, pKFn.mnId)) in sInsertedEdges:
                            continue

                        if pKFn in NonCorrectedSim3:
                            Snw = NonCorrectedSim3[pKFn]
                        else:
                            Snw = vScw[pKFn.mnId]

                        Sni = Snw * Swi

                        edge = g2o.EdgeSim3()
                        edge.set_vertex(1, optimizer.vertex(pKFn.mnId))
                        edge.set_vertex(0, optimizer.vertex(nIDi))
                        edge.set_measurement(Sni)
                        edge.set_information(matLambda)
                        optimizer.add_edge(edge)

        optimizer.initialize_optimization()
        optimizer.optimize(20)

        pMap.mMutexMapUpdate.acquire()

        for pKF in vpKFs:
            nIDi = pKF.mnId
            vSim3 = optimizer.vertex(nIDi)
            CorrectedSiw = vSim3.estimate()
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse()

            eigR = CorrectedSiw.rotation().matrix()
            eigT = CorrectedSiw.translation()
            s = CorrectedSiw.scale()

            eigT *= (1.0 / s)
            Tiw = self.converter.RT_to_TF(eigR, eigT)
            pKF.set_pose(Tiw)

        for pMP in vpMPs:
            if pMP.is_bad():
                continue

            if pMP.mnCorrectedByKF == pCurKF.mnId:
                nIDr = pMP.mnCorrectedReference
            else:
                pRefKF = pMP.get_reference_key_frame()
                nIDr = pRefKF.mnId

            Srw = vScw[nIDr]
            correctedSwr = vCorrectedSwc[nIDr]

            P3Dw = pMP.get_world_pos()
            eigCorrectedP3Dw = correctedSwr.map(Srw.map(P3Dw))

            pMP.set_world_pos(np.expand_dims(eigCorrectedP3Dw, axis=1))
            pMP.update_normal_and_depth()

        pMap.mMutexMapUpdate.release()
