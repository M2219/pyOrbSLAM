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
        vpMapPointMatches = [None] * frame.N

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
                        if vpMapPointsKF[realIdxKF]:
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

                            if vpMapPointMatches[realIdxF]:
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
                    vpMapPointMatches[idx] = None
                    nMatches -= 1

        return nMatches, vpMapPointMatches

    def search_by_BoW_kf_kf(self, pKF1, pKF2):
        vKeysUn1 = pKF1.mvKeysUn
        vFeatVec1 = pKF1.mFeatVec

        vpMapPoints1 = pKF1.get_map_point_matches()
        Descriptors1 = pKF1.mDescriptors

        vKeysUn2 = pKF2.mvKeysUn
        vFeatVec2 = pKF2.mFeatVec
        vpMapPoints2 = pKF2.get_map_point_matches()
        Descriptors2 = pKF2.mDescriptors

        vpMatches12 = [None] * len(vpMapPoints1)
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
                        if not pMP1 or pMP1.is_bad():
                            continue

                        d1 = Descriptors1[idx1]

                        bestDist1 = 256
                        bestIdx2 = -1
                        bestDist2 = 256

                        for idx2 in f2[1]:
                            pMP2 = vpMapPoints2[idx2]
                            if vbMatched2[idx2] or not pMP2 or pMP2.is_bad():
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

        return nmatches, vpMatches12


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
                if frame.mvpMapPoints[idx]:
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
        tcw = current_frame.mTcw[:3, 3:4]

        twc = -Rcw.T @ tcw

        Rlw = last_frame.mTcw[:3, :3]
        tlw = last_frame.mTcw[:3, 3:4]

        tlc = Rlw @ twc + tlw

        b_forward = tlc[2] > current_frame.mb
        b_backward = -tlc[2] > current_frame.mb

        for i in range(last_frame.N):
            pMP = last_frame.mvpMapPoints[i]
            if pMP:
                if not last_frame.mvbOutlier[i]:

                    # Project
                    x3Dw = pMP.get_world_pos()
                    x3Dc = Rcw @ x3Dw +  tcw #np.expand_dims(tcw, axis=0).T
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
                        if current_frame.mvpMapPoints[i2]:
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
                        current_frame.mvpMapPoints[idx] = None
                        n_matches -= 1

        return n_matches

    def fuse_kf_scw_mp(self, pKF, Scw, vpPoints, th, vpReplacePoint):
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
        tcw = Scw[:3, 3:4] / scw
        Ow = -Rcw.T @ tcw

        # Set of MapPoints already found in the KeyFrame
        spAlreadyFound = pKF.get_map_points()

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
            p3Dc = Rcw @ p3Dw + tcw #np.expand_dims(tcw, axis=1)

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
            PO = p3Dw - Ow #np.expand_dims(Ow, axis=1)
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

        return nFused, vpReplacePoint

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
        for pMP in vpMapPoints:

            if not pMP:
                continue

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

    def search_by_sim3(self, pKF1, pKF2, vpMatches12, s12, R12, t12, th):
        """
        Find correspondences between keyframes using the Sim3 transformation.

        Parameters:
        - pKF1, pKF2: Keyframes
        - vpMatches12: Matches between keyframes
        - s12: Scale factor
        - R12, t12: Rotation and translation between keyframes
        - th: Search radius multiplier
        """
        fx, fy = pKF1.fx, pKF1.fy
        cx, cy = pKF1.cx, pKF1.cy

        # Camera 1 and Camera 2 poses
        R1w, t1w = pKF1.get_rotation(), pKF1.get_translation()
        R2w, t2w = pKF2.get_rotation(), pKF2.get_translation()

        # Transformation between cameras
        sR12 = s12 * R12
        sR21 = (1.0 / s12) * R12.T
        t21 = -sR21 @ t12

        vpMapPoints1 = pKF1.get_map_point_matches()
        vpMapPoints2 = pKF2.get_map_point_matches()

        N1, N2 = len(vpMapPoints1), len(vpMapPoints2)

        vbAlreadyMatched1 = [False] * N1
        vbAlreadyMatched2 = [False] * N2

        for i, pMP in enumerate(vpMatches12):
            if pMP:
                vbAlreadyMatched1[i] = True
                idx2 = pMP.get_index_in_keyframe(pKF2)
                if 0 <= idx2 < N2:
                    vbAlreadyMatched2[idx2] = True

        vnMatch1 = [-1] * N1
        vnMatch2 = [-1] * N2

        # Transform from KF1 to KF2 and search
        for i1, pMP in enumerate(vpMapPoints1):
            if not pMP or vbAlreadyMatched1[i1] or pMP.is_bad():
                continue

            p3Dw = pMP.get_world_pos()
            p3Dc1 = R1w @ p3Dw + t1w
            p3Dc2 = sR21 @ p3Dc1 + t21

            if p3Dc2[2] < 0.0:
                continue

            invz = 1.0 / p3Dc2[2]
            x, y = p3Dc2[0] * invz, p3Dc2[1] * invz
            u, v = fx * x + cx, fy * y + cy

            if not pKF2.is_in_image(u, v):
                continue

            maxDistance = pMP.get_max_distance_invariance()
            minDistance = pMP.get_min_distance_invariance()
            dist3D = np.linalg.norm(p3Dc2)

            if not (minDistance <= dist3D <= maxDistance):
                continue

            nPredictedLevel = pMP.predict_scale(dist3D, pKF2)
            radius = th * pKF2.mvScaleFactors[nPredictedLevel]

            vIndices = pKF2.get_features_in_area(u, v, radius)

            if not vIndices:
                continue

            dMP = pMP.get_descriptor()
            bestDist, bestIdx = np.inf, -1
            for idx in vIndices:
                kp = pKF2.mvKeysUn[idx]
                if not (nPredictedLevel - 1 <= kp.octave <= nPredictedLevel):
                    continue

                dKF = pKF2.mDescriptors[idx]
                dist = self.descriptor_distance(dMP, dKF)

                if dist < bestDist:
                    bestDist, bestIdx = dist, idx

            if bestDist <= TH_HIGH:
                vnMatch1[i1] = bestIdx

        # Transform from KF2 to KF1 and search
        for i2, pMP in enumerate(vpMapPoints2):
            if not pMP or vbAlreadyMatched2[i2] or pMP.is_bad():
                continue

            p3Dw = pMP.get_world_pos()
            p3Dc2 = R2w @ p3Dw + t2w
            p3Dc1 = sR12 @ p3Dc2 + t12

            if p3Dc1[2] < 0.0:
                continue

            invz = 1.0 / p3Dc1[2]
            x, y = p3Dc1[0] * invz, p3Dc1[1] * invz
            u, v = fx * x + cx, fy * y + cy

            if not pKF1.is_in_image(u, v):
                continue

            maxDistance = pMP.get_max_distance_invariance()
            minDistance = pMP.get_min_distance_invariance()
            dist3D = np.linalg.norm(p3Dc1)

            if not (minDistance <= dist3D <= maxDistance):
                continue

            nPredictedLevel = pMP.predict_scale(dist3D, pKF1)
            radius = th * pKF1.mvScaleFactors[nPredictedLevel]

            vIndices = pKF1.get_features_in_area(u, v, radius)

            if not vIndices:
                continue

            dMP = pMP.get_descriptor()
            bestDist, bestIdx = np.inf, -1
            for idx in vIndices:
                kp = pKF1.mvKeysUn[idx]
                if not (nPredictedLevel - 1 <= kp.octave <= nPredictedLevel):
                    continue

                dKF = pKF1.mDescriptors[idx]
                dist = self.descriptor_distance(dMP, dKF)

                if dist < bestDist:
                    bestDist, bestIdx = dist, idx

            if bestDist <= TH_HIGH:
                vnMatch2[i2] = bestIdx

        # Check agreement
        nFound = 0
        for i1, idx2 in enumerate(vnMatch1):
            if idx2 >= 0:
                idx1 = vnMatch2[idx2]
                if idx1 == i1:
                    vpMatches12[i1] = vpMapPoints2[idx2]
                    nFound += 1

        return nFound, vpMatches12

    def search_by_projection_ckf_scw_mp(self, pKF, Scw, vpPoints, vpMatched, th):
        """
        Projects map points into the keyframe and finds matches.

        Parameters:
        - pKF: KeyFrame object with calibration parameters and features.
        - Scw: 4x4 transformation matrix (Sim3).
        - vpPoints: List of candidate MapPoints.
        - vpMatched: List of matched MapPoints (to be updated).
        - th: Search radius multiplier.

        Returns:
        - nmatches: Number of matches found.
        """
        # Get Calibration Parameters
        fx, fy, cx, cy = pKF.fx, pKF.fy, pKF.cx, pKF.cy

        # Decompose Scw
        sRcw = Scw[:3, :3]
        scw = np.linalg.norm(sRcw[0])
        Rcw = sRcw / scw
        tcw = Scw[:3, 3:4] / scw
        Ow = -Rcw.T @ tcw

        # Set of MapPoints already found in the KeyFrame
        spAlreadyFound = set(vpMatched) - {None}

        nmatches = 0

        # For each Candidate MapPoint Project and Match
        for iMP, pMP in enumerate(vpPoints):
            # Discard bad MapPoints and already found
            if pMP.is_bad() or pMP in spAlreadyFound:
                continue

            # Get 3D Coords
            p3Dw = pMP.get_world_pos()

            # Transform into Camera Coords
            p3Dc = Rcw @ p3Dw +  tcw #np.expand_dims(tcw, axis=0).T
            # Depth must be positive
            if p3Dc[2][0] <= 0.0:
                continue

            # Project into Image
            invz = 1.0 / p3Dc[2]
            x, y = p3Dc[0] * invz, p3Dc[1] * invz
            u, v = fx * x + cx, fy * y + cy

            # Point must be inside the image
            if not pKF.is_in_image(u, v):
                continue

            # Depth must be inside the scale invariance region of the point
            maxDistance = pMP.get_max_distance_invariance()
            minDistance = pMP.get_min_distance_invariance()
            PO = p3Dw -  Ow #np.expand_dims(Ow, axis=0).T
            dist = np.linalg.norm(PO)

            if dist < minDistance or dist > maxDistance:
                continue

            # Viewing angle must be less than 60 degrees
            Pn = pMP.get_normal()

            if np.dot(PO.T, Pn) < 0.5 * dist:
                continue

            # Predicted scale level
            nPredictedLevel = pMP.predict_scale(dist, pKF)

            # Search in a radius
            radius = th * pKF.mvScaleFactors[nPredictedLevel]
            vIndices = pKF.get_features_in_area(u, v, radius)

            if not vIndices:
                continue

            # Match to the most similar keypoint in the radius
            dMP = pMP.get_descriptor()
            bestDist, bestIdx = 256, -1

            for idx in vIndices:
                if vpMatched[idx] is not None:
                    continue

                kpLevel = pKF.mvKeysUn[idx].octave
                if kpLevel < nPredictedLevel - 1 or kpLevel > nPredictedLevel:
                    continue

                dKF = pKF.mDescriptors[idx]
                dist = self.descriptor_distance(dMP, dKF)

                if dist < bestDist:
                    bestDist = dist
                    bestIdx = idx

            if bestDist <= TH_LOW:
                vpMatched[bestIdx] = pMP
                nmatches += 1

        return nmatches, vpMatched

    def search_by_projection_f_kf_f(self, CurrentFrame, pKF, sAlreadyFound, th, ORBdist):
        nmatches = 0

        Rcw = CurrentFrame.mTcw[:3, :3]
        tcw = CurrentFrame.mTcw[:3, 3:4]
        Ow = -np.dot(Rcw.T, tcw)

        # Rotation Histogram (to check rotation consistency)
        rotHist = [[] for _ in range(HISTO_LENGTH)]
        factor = 1.0 / HISTO_LENGTH

        vpMPs = pKF.get_map_point_matches()

        for i, pMP in enumerate(vpMPs):
            if pMP and not pMP.is_bad() and pMP not in sAlreadyFound:
                # Project
                x3Dw = pMP.get_world_pos()
                x3Dc = np.dot(Rcw, x3Dw) + tcw

                xc = x3Dc[0]
                yc = x3Dc[1]
                invzc = 1.0 / x3Dc[2]

                u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx
                v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy

                if u < CurrentFrame.mnMinX or u > CurrentFrame.mnMaxX or v < CurrentFrame.mnMinY or v > CurrentFrame.mnMaxY:
                    continue

                # Compute predicted scale level
                PO = x3Dw - Ow
                dist3D = np.linalg.norm(PO)

                maxDistance = pMP.get_max_distance_invariance()
                minDistance = pMP.get_min_distance_invariance()

                if dist3D < minDistance or dist3D > maxDistance:
                    continue

                nPredictedLevel = pMP.predict_scale(dist3D, CurrentFrame)

                # Search in a window
                radius = th * CurrentFrame.mvScaleFactors[nPredictedLevel]
                vIndices2 = CurrentFrame.get_features_in_area(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1)

                if not vIndices2:
                    continue

                dMP = pMP.get_descriptor()

                bestDist = 256
                bestIdx2 = -1

                for i2 in vIndices2:
                    if CurrentFrame.mvpMapPoints[i2]:
                        continue

                    d = CurrentFrame.mDescriptors[i2]
                    dist = self.descriptor_distance(dMP, d)

                    if dist < bestDist:
                        bestDist = dist
                        bestIdx2 = i2

                if bestDist <= ORBdist:
                    CurrentFrame.mvpMapPoints[bestIdx2] = pMP
                    nmatches += 1

                    if self.mbCheckOrientation:
                        rot = pKF.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle
                        if rot < 0.0:
                            rot += 360.0
                        bin = round(rot * factor)
                        if bin == HISTO_LENGTH:
                            bin = 0
                        assert 0 <= bin < HISTO_LENGTH
                        rotHist[bin].append(bestIdx2)

        # Enforce rotation consistency
        if self.mbCheckOrientation:
            ind1, ind2, ind3 = self.compute_three_maxima(rotHist, HISTO_LENGTH)

            for i in range(HISTO_LENGTH):
                if i != ind1 and i != ind2 and i != ind3:
                    for idx in rotHist[i]:
                        CurrentFrame.mvpMapPoints[idx] = None
                        nmatches -= 1

        return nmatches

if __name__ == "__main__":

    ORb = ORBMatcher(0.5, 0.5)
    a = np.array([3, 191, 24, 185, 182, 169, 189, 31, 30, 9, 90, 231, 181, 10, 192, 0, 183, 27, 31, 149, 147, 152, 69, 127, 172, 4, 62, 192, 156, 72, 129, 153])
    b = np.array([90, 215, 107, 14, 210, 82, 63, 189, 49, 76, 223, 231, 73, 102, 158, 213, 54, 215, 237, 230, 253, 154, 173, 125, 50, 99, 170, 196, 47, 166, 175, 85])
    #print(ORb.distance(a, b))
