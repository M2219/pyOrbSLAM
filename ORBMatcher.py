import numpy as np
TH_HIGH = 100
TH_LOW = 50
HISTO_LENGTH = 30


class ORBMatcher:
    def __init__(self, nnratio, checkOri):

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

    def search_by_BoW(self, kf, frame, vpMapPointMatches):
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
                    vpMapPointMatches[idx] = None
                    nMatches -= 1

        return nMatches

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



if __name__ == "__main__":

    ORb = ORBMatcher(0.5, 0.5)
    a = np.array([3, 191, 24, 185, 182, 169, 189, 31, 30, 9, 90, 231, 181, 10, 192, 0, 183, 27, 31, 149, 147, 152, 69, 127, 172, 4, 62, 192, 156, 72, 129, 153])
    b = np.array([90, 215, 107, 14, 210, 82, 63, 189, 49, 76, 223, 231, 73, 102, 158, 213, 54, 215, 237, 230, 253, 154, 173, 125, 50, 99, 170, 196, 47, 166, 175, 85])
    #print(ORb.distance(a, b))
