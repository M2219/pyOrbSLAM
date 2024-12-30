import numpy as np
TH_HIGH = 100
TH_LOW = 50
HISTO_LENGTH = 30


class ORBMatcher:
    def __init__(self, nnratio=1, checkOri=True):

        self.mfNNratio = nnratio
        self.mbCheckOrientation = checkOri

    def descriptor_distance(self, a, b):
        xor = np.bitwise_xor(a, b)
        return sum(bin(byte).count('1') for byte in xor)

    def compute_three_maxima(self, histo, histo_length):
        # Compute indices of the three largest histogram bins
        histo_counts = [len(h) for h in histo]
        indices = np.argsort(histo_counts)[::-1][:3]  # Top 3 indices
        return indices

    def search_by_BoW_kf_f(self, kf, frame):
        vpMapPointsKF = kf.get_map_point_matches()

        vpMapPointMatches = {}

        vFeatVecKF = kf.mFeatVec

        nMatches = 0

        rotHist = [[] for _ in range(HISTO_LENGTH)]
        factor = 1.0 / HISTO_LENGTH

        # Matching over ORB features in the same vocabulary node
        #print(vFeatVecKF.keys())
        #print(frame.mFeatVec.keys())

        vk = list(vFeatVecKF.values())
        vf = list(frame.mFeatVec.values())

        #for k, v in zip(vk, vf):
        #    print("len = ", len(k) , len(v))
        #exit()

        KFit = iter(vFeatVecKF)
        Fit = iter(frame.mFeatVec)

        try:
            KFitEntry = next(KFit)
            FitEntry = next(Fit)
            while True:
                if KFitEntry == FitEntry:
                    vIndicesKF = vFeatVecKF[KFitEntry]
                    vIndicesF = frame.mFeatVec[FitEntry]

                    for idxKF in vIndicesKF:
                        realIdxKF = idxKF
                        if realIdxKF in vpMapPointsKF:
                            pMP = vpMapPointsKF[realIdxKF]
                        else:
                            continue

                        if pMP.is_bad():
                            continue

                        dKF = kf.mDescriptors[realIdxKF]

                        bestDist1 = 256
                        bestIdxF = -1
                        bestDist2 = 256

                        for idxF in vIndicesF:
                            realIdxF = idxF

                            if realIdxF in vpMapPointMatches:
                                continue

                            dF = frame.mDescriptors[realIdxF]
                            dist = self.descriptor_distance(dKF, dF)

                            if dist < bestDist1:
                                bestDist2 = bestDist1
                                bestDist1 = dist
                                bestIdxF = realIdxF

                            elif dist < bestDist2:
                                bestDist2 = dist

                        if bestDist1 <= TH_LOW:
                            if float(bestDist1) < self.mfNNratio * float(bestDist2):
                                vpMapPointMatches[bestIdxF] = pMP

                                kp = kf.mvKeysUn[realIdxKF]

                                if self.mbCheckOrientation:
                                    rot = kp.angle - frame.mvKeys[bestIdxF].angle
                                    if rot < 0.0:
                                        rot += 360.0
                                    binIdx = round(rot * factor)
                                    if binIdx == HISTO_LENGTH:
                                        binIdx = 0
                                    assert 0 <= binIdx < HISTO_LENGTH
                                    rotHist[binIdx].append(bestIdxF)

                                nMatches += 1

                    KFitEntry = next(KFit)
                    FitEntry = next(Fit)

                elif KFitEntry < FitEntry:
                    KFitEntry = next(KFit)
                else:
                    FitEntry = next(Fit)

        except StopIteration:
            pass

        if self.mbCheckOrientation:
            ind1, ind2, ind3 = self.compute_three_maxima(rotHist, HISTO_LENGTH)

            for i in range(HISTO_LENGTH):
                if i in [ind1, ind2, ind3]:
                    continue
                for idx in rotHist[i]:
                    del vpMapPointMatches[idx]
                    nMatches -= 1

        return nMatches, vpMapPointMatches

    def search_by_BoW_kf_kf(self, pKF1, pKF2, vpMatches12):
        vKeysUn1 = pKF1.mvKeysUn
        vFeatVec1 = pKF1.mFeatVec

        vpMapPoints1 = pKF1.get_map_point_matches()
        Descriptors1 = pKF1.mDescriptors

        vKeysUn2 = pKF2.mvKeysUn
        vFeatVec2 = pKF2.mFeatVec
        vpMapPoints2 = pKF2.get_map_point_matches()
        Descriptors2 = pKF2.mDescriptors

        vpMatches12[:] = [None] * len(vpMapPoints1)
        vbMatched2 = [False] * len(vpMapPoints2)

        rotHist = [[] for _ in range(HISTO_LENGTH)]


        factor = 1.0 / HISTO_LENGTH

        nmatches = 0

        f1it = iter(vFeatVec1.items())
        f2it = iter(vFeatVec2.items())
        f1end = len(vFeatVec1)
        f2end = len(vFeatVec2)

        try:
            f1 = next(f1it)
            f2 = next(f2it)
            while True:
                if f1[0] == f2[0]:
                    for idx1 in f1[1]:
                        pMP1 = vpMapPoints1[idx1]
                        if not pMP1 or pMP1.isBad():
                            continue

                        d1 = Descriptors1[idx1]

                        bestDist1 = 256
                        bestIdx2 = -1
                        bestDist2 = 256

                        for idx2 in f2[1]:
                            pMP2 = vpMapPoints2[idx2]
                            if vbMatched2[idx2] or not pMP2 or pMP2.isBad():
                                continue

                            d2 = Descriptors2[idx2]
                            dist = self.descriptor_distance(d1, d2)

                            if dist < bestDist1:
                                bestDist2 = bestDist1
                                bestDist1 = dist
                                bestIdx2 = idx2
                            elif dist < bestDist2:
                                bestDist2 = dist

                        if bestDist1 < TH_LOW:
                            if bestDist1 < self.mfNNratio * bestDist2:
                                vpMatches12[idx1] = vpMapPoints2[bestIdx2]
                                vbMatched2[bestIdx2] = True

                                if self.mbCheckOrientation:
                                    rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle
                                    if rot < 0.0:
                                        rot += 360.0
                                    bin_idx = round(rot * factor)
                                    if bin_idx == HISTO_LENGTH:
                                        bin_idx = 0
                                    rotHist[bin_idx].append(idx1)

                                nmatches += 1
                    f1 = next(f1it)
                    f2 = next(f2it)
                elif f1[0] < f2[0]:
                    f1 = next(f1it)
                else:
                    f2 = next(f2it)
        except StopIteration:
            pass

        if self.mbCheckOrientation:
            ind1, ind2, ind3 = self.compute_three_maxima(rotHist, HISTO_LENGTH)

            for i in range(HISTO_LENGTH):
                if i in [ind1, ind2, ind3]:
                    continue
                for idx in rotHist[i]:
                    vpMatches12[idx] = None
                    nmatches -= 1

        return nmatches


    def search_by_projection_f_p(self, frame, vp_map_points, th):
        """
        Perform projection-based search to associate MapPoints with the current frame's keypoints.

        Args:
            frame: The current frame object.
            vp_map_points: List of MapPoint objects to search for.
            th (float): The threshold for the search radius.

        Returns:
            int: The number of matches found.
        """
        n_matches = 0
        b_factor = th != 1.0

        for pMP in vp_map_points:
            if not pMP.mbTrackInView:
                continue

            if pMP.is_bad():
                continue

            n_predicted_level = pMP.mnTrackScaleLevel

            # Determine the search radius based on the viewing direction
            r = self.radius_by_viewing_cos(pMP.mTrackViewCos)

            if b_factor:
                r *= th

            # Search for features in the area
            v_indices = frame.get_features_in_area(
                pMP.mTrackProjX,
                pMP.mTrackProjY,
                r * frame.mvScaleFactors[n_predicted_level],
                n_predicted_level - 1,
                n_predicted_level
            )

            if not v_indices:
                continue

            MPdescriptor = pMP.get_descriptor()

            best_dist = 256
            best_level = -1
            best_dist2 = 256
            best_level2 = -1
            best_idx = -1

            # Find the best and second-best matches
            for idx in v_indices:
                if (idx in frame.mvpMapPoints):
                    if frame.mvpMapPoints[idx].observations() > 0:
                        continue

                if frame.mvuRight[idx] > 0:
                    er = abs(pMP.mTrackProjXR - frame.mvuRight[idx])
                    if er > r * frame.mvScaleFactors[n_predicted_level]:
                        continue

                d = frame.mDescriptors[idx]

                dist = self.descriptor_distance(MPdescriptor, d)

                if dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = dist
                    best_level2 = best_level
                    best_level = frame.mvKeysUn[idx].octave
                    best_idx = idx
                elif dist < best_dist2:
                    best_level2 = frame.mvKeysUn[idx].octave
                    best_dist2 = dist

            # Apply ratio test (only if best and second-best are in the same scale level)
            if best_dist <= TH_HIGH:
                if best_level == best_level2 and best_dist > self.mfNNratio * best_dist2:
                    continue

                frame.mvpMapPoints[best_idx] = pMP
                frame.mvbOutlier[best_idx] = False
                n_matches += 1

        return n_matches

    def radius_by_viewing_cos(self, view_cos):
        """
        Calculate the search radius based on the viewing cosine value.

        Args:
            view_cos (float): The viewing cosine value.

        Returns:
            float: The calculated radius.
        """
        if view_cos > 0.998:
            return 2.5
        else:
            return 4.0

    def search_by_projection_f_f(self, current_frame, last_frame, th):
        """
        Perform projection-based search to find correspondences between the current and last frames.

        Args:
            current_frame: The current frame object.
            last_frame: The last frame object.
            th (float): The threshold for searching in a window.
            b_mono (bool): Flag indicating whether the system is monocular.

        Returns:
            int: The number of matches found.
        """

        n_matches = 0

        # Rotation Histogram (for rotation consistency check)
        rot_hist = [[] for _ in range(HISTO_LENGTH)]
        factor = 1.0 / HISTO_LENGTH

        Rcw = current_frame.mTcw[:3, :3]
        tcw = current_frame.mTcw[:3, 3]

        twc = -Rcw.T @ tcw

        Rlw = last_frame.mTcw[:3, :3]
        tlw = last_frame.mTcw[:3, 3]

        tlc = Rlw @ twc + tlw

        b_forward = tlc[2] > current_frame.mb
        b_backward = -tlc[2] > current_frame.mb

        for i in list(last_frame.mvpMapPoints.keys()):
            if i in last_frame.mvpMapPoints:
                pMP = last_frame.mvpMapPoints[i]
                if not last_frame.mvbOutlier[i]:
                    # Project
                    x3Dw = pMP.get_world_pos()
                    x3Dc = Rcw @ x3Dw + np.expand_dims(tcw, axis=0).T
                    xc, yc, zc = x3Dc[0][0], x3Dc[1][0], x3Dc[2][0]
                    invzc = 1.0 / zc

                    if invzc < 0:
                        continue

                    u = current_frame.fx * xc * invzc + current_frame.cx
                    v = current_frame.fy * yc * invzc + current_frame.cy

                    if u < current_frame.mnMinX or u > current_frame.mnMaxX:
                        continue
                    if v < current_frame.mnMinY or v > current_frame.mnMaxY:
                        continue

                    n_last_octave = last_frame.mvKeys[i].octave


                    # Search in a window. Size depends on scale
                    radius = th * current_frame.mvScaleFactors[n_last_octave]
                    #print(b_forward, b_backward)
                    if b_forward:
                        v_indices2 = current_frame.get_features_in_area(u, v, radius, n_last_octave, -1)

                    elif b_backward:
                        v_indices2 = current_frame.get_features_in_area(u, v, radius, 0, n_last_octave)

                    else:
                        v_indices2 = current_frame.get_features_in_area(u, v, radius, n_last_octave - 1, n_last_octave + 1)

                    if not v_indices2:
                        continue

                    dMP = pMP.get_descriptor()

                    best_dist = 256
                    best_idx2 = -1

                    for i2 in v_indices2:
                        if i2 in current_frame.mvpMapPoints:
                            if current_frame.mvpMapPoints[i2].observations() > 0:
                               continue

                        if current_frame.mvuRight[i2] > 0:
                            ur = u - current_frame.mbf * invzc
                            er = abs(ur - current_frame.mvuRight[i2])
                            if er > radius:
                                continue

                        d = current_frame.mDescriptors[i2]
                        dist = self.descriptor_distance(dMP, d)

                        if dist < best_dist:
                            best_dist = dist
                            best_idx2 = i2


                    if best_dist <= TH_HIGH:
                        current_frame.mvpMapPoints[best_idx2] = pMP
                        current_frame.mvbOutlier[best_idx2] = False
                        n_matches += 1

                        if self.mbCheckOrientation:
                            rot = last_frame.mvKeysUn[i].angle - current_frame.mvKeysUn[best_idx2].angle
                            if rot < 0.0:
                                rot += 360.0
                            bin_idx = round(rot * factor)
                            if bin_idx == HISTO_LENGTH:
                                bin_idx = 0
                            assert 0 <= bin_idx < HISTO_LENGTH
                            rot_hist[bin_idx].append(best_idx2)

        # Apply rotation consistency
        if self.mbCheckOrientation:
            ind1, ind2, ind3 = self.compute_three_maxima(rot_hist, HISTO_LENGTH)

            for i in range(HISTO_LENGTH):
                if i not in (ind1, ind2, ind3):
                    for idx in rot_hist[i]:
                        del current_frame.mvpMapPoints[idx]
                        del current_frame.mvbOutlier[idx]
                        n_matches -= 1

        return n_matches

    def fuse(self, pKF, Scw, vpPoints, th, vpReplacePoint):
        """
        Fuse map points into a keyframe by projecting and matching them.

        Args:
            pKF: The keyframe object.
            Scw (np.ndarray): The transformation matrix (4x4).
            vpPoints (list): List of candidate MapPoint objects.
            th (float): Radius threshold for matching.
            vpReplacePoint (list): List to store points that should be replaced.

        Returns:
            int: The number of fused points.
        """
        # Get calibration parameters
        fx, fy, cx, cy = pKF.fx, pKF.fy, pKF.cx, pKF.cy

        # Decompose Scw
        sRcw = Scw[:3, :3]
        scw = np.sqrt(np.dot(sRcw[0], sRcw[0]))
        Rcw = sRcw / scw
        tcw = Scw[:3, 3] / scw
        Ow = -Rcw.T @ tcw

        # Set of MapPoints already found in the KeyFrame
        spAlreadyFound = set(pKF.get_map_points())

        nFused = 0
        nPoints = len(vpPoints)

        # Process each candidate MapPoint
        for iMP in range(nPoints):
            pMP = vpPoints[iMP]

            # Discard bad MapPoints and already found points
            if pMP.is_bad() or pMP in spAlreadyFound:
                continue

            # Get 3D world coordinates
            p3Dw = pMP.get_world_pos()

            # Transform into camera coordinates
            p3Dc = Rcw @ p3Dw + np.expand_dims(tcw, axis=0).T

            # Depth must be positive
            if p3Dc[2][0] < 0.0:
                continue

            # Project into image
            invz = 1.0 / p3Dc[2][0]
            x = p3Dc[0][0] * invz
            y = p3Dc[1][0] * invz
            u = fx * x + cx
            v = fy * y + cy

            # Check if point is inside the image
            if not pKF.is_in_image(u, v):
                continue

            # Check depth constraints
            maxDistance = pMP.get_max_distance_invariance()
            minDistance = pMP.get_min_distance_invariance()
            PO = p3Dw - Ow
            dist3D = np.linalg.norm(PO)

            if dist3D < minDistance or dist3D > maxDistance:
                continue

            # Check viewing angle
            Pn = pMP.get_normal()
            if np.dot(PO, Pn) < 0.5 * dist3D:
                continue

            # Predict scale level
            nPredictedLevel = pMP.predict_scale(dist3D, pKF)

            # Search in a radius
            radius = th * pKF.mvScaleFactors[nPredictedLevel]
            vIndices = pKF.get_features_in_area(u, v, radius)

            if not vIndices:
                continue

            # Match to the most similar keypoint in the radius
            dMP = pMP.get_descriptor()

            bestDist = float('inf')
            bestIdx = -1

            for idx in vIndices:
                kpLevel = pKF.mvKeysUn[idx].octave

                if kpLevel < nPredictedLevel - 1 or kpLevel > nPredictedLevel:
                    continue

                dKF = pKF.mDescriptors[idx]

                dist = self.descriptor_distance(dMP, dKF)

                if dist < bestDist:
                    bestDist = dist
                    bestIdx = idx

            # If there is already a MapPoint, replace; otherwise, add a new measurement
            if bestDist <= TH_LOW:
                pMPinKF = pKF.get_map_point(bestIdx)
                if pMPinKF:
                    if not pMPinKF.is_bad():
                        vpReplacePoint[iMP] = pMPinKF
                else:
                    pMP.add_observation(pKF, bestIdx)
                    pKF.add_map_point(pMP, bestIdx)
                nFused += 1

        return nFused

    def fuse_pkf_mp(self, pKF, vpMapPoints, th):
        """
        Fuse map points into a keyframe by projecting and matching them.

        Args:
            pKF: The keyframe object.
            vpMapPoints (list): List of MapPoint objects to be fused.
            th (float): Radius threshold for matching.

        Returns:
            int: The number of fused points.
        """
        # Get calibration parameters
        fx, fy, cx, cy, bf = pKF.fx, pKF.fy, pKF.cx, pKF.cy, pKF.mbf

        # Get camera pose and center
        Rcw = pKF.get_rotation()
        tcw = pKF.get_translation()
        Ow = pKF.get_camera_center()

        nFused = 0
        for i in range(len(vpMapPoints)):
            pMP = vpMapPoints[i]
            if pMP.is_bad() or pMP.is_in_key_frame(pKF):
                continue

            # Get 3D world coordinates
            p3Dw = pMP.get_world_pos()

            # Transform into camera coordinates
            p3Dc = Rcw @ p3Dw + tcw

            # Depth must be positive
            if p3Dc[2][0] < 0.0:
                continue

            invz = 1.0 / p3Dc[2][0]
            x = p3Dc[0][0] * invz
            y = p3Dc[1][0] * invz

            u = fx * x + cx
            v = fy * y + cy
            ur = u - bf * invz

            # Check if point is inside the image
            if not pKF.is_in_image(u, v):
                continue

            # Check depth constraints
            maxDistance = pMP.get_max_distance_invariance()
            minDistance = pMP.get_min_distance_invariance()
            PO = p3Dw - Ow
            dist3D = np.linalg.norm(PO)

            if dist3D < minDistance or dist3D > maxDistance:
                continue

            # Check viewing angle
            Pn = pMP.get_normal()
            if np.dot(PO.T, Pn) < 0.5 * dist3D:
                continue

            # Predict scale level
            nPredictedLevel = pMP.predict_scale(dist3D, pKF)

            # Search in a radius
            radius = th * pKF.mvScaleFactors[nPredictedLevel]
            vIndices = pKF.get_features_in_area(u, v, radius)

            if not vIndices:
                continue

            # Match to the most similar keypoint in the radius
            dMP = pMP.get_descriptor()

            bestDist = float('inf')
            bestIdx = -1

            for idx in vIndices:
                kp = pKF.mvKeysUn[idx]
                kpLevel = kp.octave

                if kpLevel < nPredictedLevel - 1 or kpLevel > nPredictedLevel:
                    continue

                if pKF.mvuRight[idx] >= 0:
                    # Check reprojection error in stereo
                    kpx, kpy, kpr = kp.pt[0], kp.pt[1], pKF.mvuRight[idx]
                    ex, ey, er = u - kpx, v - kpy, ur - kpr
                    e2 = ex ** 2 + ey ** 2 + er ** 2

                    if e2 * pKF.mvInvLevelSigma2[kpLevel] > 7.8:
                        continue
                else:
                    # Check reprojection error in monocular
                    kpx, kpy = kp.pt[0], kp.pt[1]
                    ex, ey = u - kpx, v - kpy
                    e2 = ex ** 2 + ey ** 2

                    if e2 * pKF.mvInvLevelSigma2[kpLevel] > 5.99:
                        continue

                dKF = pKF.mDescriptors[idx]
                dist = self.descriptor_distance(dMP, dKF)

                if dist < bestDist:
                    bestDist = dist
                    bestIdx = idx

            # If there is already a MapPoint, replace; otherwise, add a new measurement
            if bestDist <= TH_LOW:
                pMPinKF = pKF.get_map_point(bestIdx)
                if pMPinKF:
                    if not pMPinKF.is_bad():
                        if pMPinKF.observations() > pMP.observations():
                            pMP.replace(pMPinKF)
                        else:
                            pMPinKF.replace(pMP)
                else:
                    pMP.add_observation(pKF, bestIdx)
                    pKF.add_map_point(pMP, bestIdx)
                nFused += 1

        return nFused

    def search_for_triangulation(self, pKF1, pKF2, F12, bOnlyStereo=False):

        vFeatVec1 = pKF1.mFeatVec
        vFeatVec2 = pKF2.mFeatVec

        # Compute epipole in the second image
        Cw = pKF1.get_camera_center()
        R2w = pKF2.get_rotation()
        t2w = pKF2.get_translation()
        C2 = R2w @ Cw + t2w
        invz = 1.0 / C2[2]
        ex = pKF2.fx * C2[0] * invz + pKF2.cx
        ey = pKF2.fy * C2[1] * invz + pKF2.cy

        nmatches = 0
        vbMatched2 = [False] * pKF2.N
        vMatches12 = [-1] * pKF1.N

        rotHist = [[] for _ in range(HISTO_LENGTH)]
        factor = 1.0 / HISTO_LENGTH

        f1it = iter(vFeatVec1.items())
        f2it = iter(vFeatVec2.items())

        while True:
            try:
                f1key, f1val = next(f1it)
                f2key, f2val = next(f2it)
            except StopIteration:
                break

            if f1key == f2key:
                for idx1 in f1val:
                    pMP1 = pKF1.get_map_point(idx1)

                    # If there is already a MapPoint skip
                    if pMP1:
                        continue

                    bStereo1 = pKF1.mvuRight[idx1] >= 0
                    if bOnlyStereo and not bStereo1:
                        continue

                    kp1 = pKF1.mvKeysUn[idx1]
                    d1 = pKF1.mDescriptors[idx1]

                    bestDist = TH_LOW
                    bestIdx2 = -1

                    for idx2 in f2val:
                        pMP2 = pKF2.get_map_point(idx2)

                        # If already matched or there is a MapPoint, skip
                        if vbMatched2[idx2] or pMP2:
                            continue

                        bStereo2 = pKF2.mvuRight[idx2] >= 0
                        if bOnlyStereo and not bStereo2:
                            continue

                        d2 = pKF2.mDescriptors[idx2]
                        dist = self.descriptor_distance(d1, d2)

                        if dist > TH_LOW or dist > bestDist:
                            continue

                        kp2 = pKF2.mvKeysUn[idx2]

                        if not bStereo1 and not bStereo2:
                            distex = ex - kp2.pt[0]
                            distey = ey - kp2.pt[1]
                            if distex**2 + distey**2 < 100 * pKF2.mvScaleFactors[kp2.octave]:
                                continue

                        if self.check_dist_epipolar_line(kp1, kp2, F12, pKF2):
                            bestIdx2 = idx2
                            bestDist = dist

                    if bestIdx2 >= 0:
                        kp2 = pKF2.mvKeysUn[bestIdx2]
                        vMatches12[idx1] = bestIdx2
                        nmatches += 1
                        vbMatched2[bestIdx2] = True

                        if self.mbCheckOrientation:
                            rot = kp1.angle - kp2.angle
                            if rot < 0:
                                rot += 360.0
                            bin_idx = round(rot * factor)
                            if bin_idx == HISTO_LENGTH:
                                bin_idx = 0
                            rotHist[bin_idx].append(idx1)

            elif f1key < f2key:
                f1key = next(f1it)
            else:
                f2key = next(f2it)

        if self.mbCheckOrientation:
            # Find the top three maxima in the orientation histogram
            ind1, ind2, ind3 = self.compute_three_maxima(rotHist, HISTO_LENGTH)

            # Remove matches not in the top three orientation bins
            for i in range(HISTO_LENGTH):
                if i == ind1 or i == ind2 or i == ind3:
                    continue

                for j in rotHist[i]:
                    vMatches12[j] = -1
                    nmatches -= 1

        # Clear and populate the matched pairs
        vMatchedPairs = []
        for i, match in enumerate(vMatches12):
            if match < 0:
                continue
            vMatchedPairs.append((i, match))

        return vMatchedPairs

    def check_dist_epipolar_line(self, kp1, kp2, F12, pKF2):
        """
        Checks if a pair of keypoints satisfies the epipolar constraint.
        Args:
            kp1: Keypoint in the first image (with attributes `pt` as (x, y)).
            kp2: Keypoint in the second image (with attributes `pt` as (x, y)).
            F12: Fundamental matrix (3x3 NumPy array).
            pKF2: Second keyframe object with `mvLevelSigma2` (list of scale uncertainties).
        Returns:
            bool: True if the keypoints satisfy the epipolar constraint, False otherwise.
        """
        # Compute epipolar line coefficients
        a = kp1.pt[0] * F12[0, 0] + kp1.pt[1] * F12[1, 0] + F12[2, 0]
        b = kp1.pt[0] * F12[0, 1] + kp1.pt[1] * F12[1, 1] + F12[2, 1]
        c = kp1.pt[0] * F12[0, 2] + kp1.pt[1] * F12[1, 2] + F12[2, 2]

        # Compute distance from keypoint to epipolar line
        num = a * kp2.pt[0] + b * kp2.pt[1] + c
        den = a**2 + b**2
        if den == 0:
            return False

        dsqr = (num**2) / den

        # Check threshold
        threshold = 3.84 * pKF2.mvLevelSigma2[kp2.octave]
        return dsqr < threshold

if __name__ == "__main__":

    ORb = ORBMatcher(0.5, 0.5)
    a = np.array([3, 191, 24, 185, 182, 169, 189, 31, 30, 9, 90, 231, 181, 10, 192, 0, 183, 27, 31, 149, 147, 152, 69, 127, 172, 4, 62, 192, 156, 72, 129, 153])
    b = np.array([90, 215, 107, 14, 210, 82, 63, 189, 49, 76, 223, 231, 73, 102, 158, 213, 54, 215, 237, 230, 253, 154, 173, 125, 50, 99, 170, 196, 47, 166, 175, 85])
    #print(ORb.distance(a, b))
