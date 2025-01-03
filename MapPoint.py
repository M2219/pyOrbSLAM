import threading
import numpy as np

class MapPoint:
    nNextId = 0
    mGlobalMutex = threading.Lock()
    def __init__(self, Pos, pRefKF, pMap, idxF=None, kframe_bool=True):
        self.mMutexPos = threading.Lock()
        self.mMutexFeatures = threading.Lock()

        self.mpMap = pMap

        if kframe_bool:

            self.mnFirstKFid = pRefKF.mnId
            self.mnFirstFrame = pRefKF.mnFrameId
            self.mpRefKF = pRefKF

            self.mfMinDistance = 0
            self.mfMaxDistance = 0

            self.mWorldPos = Pos.copy()
            self.mNormalVector = np.zeros((3, 1), dtype=np.float32)

            self.mDescriptor = pRefKF.mDescriptors[idxF]

            self.mnId = MapPoint.nNextId
            with self.mpMap.mMutexPointCreation:
                MapPoint.nNextId += 1

        else:
            self.mnFirstKFid = -1
            self.mnFirstFrame = pRefKF.mnId
            self.mpRefKF = None

            self.mWorldPos = Pos.copy()
            Ow = pRefKF.get_camera_center()
            self.mNormalVector = self.mWorldPos - Ow
            self.mNormalVector = self.mNormalVector / np.linalg.norm(self.mNormalVector)

            PC = Pos - Ow
            dist = np.linalg.norm(PC)

            level = pRefKF.mvKeysUn[idxF].octave
            levelScaleFactor = pRefKF.mvScaleFactors[level]
            nLevels = pRefKF.mnScaleLevels

            self.mfMaxDistance = dist * levelScaleFactor
            self.mfMinDistance = self.mfMaxDistance / pRefKF.mvScaleFactors[nLevels-1]

            self.mDescriptor = pRefKF.mDescriptors[idxF]

            self.mnId = MapPoint.nNextId
            with self.mpMap.mMutexPointCreation:
                MapPoint.nNextId += 1

        self.nObs = 0
        self.mnTrackReferenceForFrame = 0
        self.mnLastFrameSeen = 0

        self.mnBALocalForKF = 0
        self.mnFuseCandidateForKF = 0
        self.mnLoopPointForKF = 0
        self.mnCorrectedByKF = 0
        self.mnCorrectedReference = 0
        self.mnBAGlobalForKF = 0

        self.mnVisible = 1
        self.mnFound = 1

        self.mbBad = False
        self.mpReplaced = None
        self.mObservations = {}
        self.mbTrackInView = None

    def descriptor_distance(self, a, b):
        xor = np.bitwise_xor(a, b)
        return sum(bin(byte).count('1') for byte in xor)

    def set_world_pos(self, Pos):

        with self.mGlobalMutex:
            with self.mMutexPos:
                self.mWorldPos = Pos.copy()

    def get_world_pos(self):
        with self.mMutexPos:
            return self.mWorldPos.copy()

    def get_normal(self):
        with self.mMutexPos:
            return self.mNormalVector.copy()

    def get_reference_key_frame(self):
        with self.mMutexFeatures:
            return self.mpRefKF

    def add_observation(self, pKF, idx):
        with self.mMutexFeatures:
            if pKF in self.mObservations:
                return

            self.mObservations[pKF] = idx
            if pKF.mvuRight[idx] >= 0:
                self.nObs += 2
            else:
                self.nObs += 1

    def erase_observation(self, pKF):
        bBad = False
        with self.mMutexFeatures:
            if pKF in self.mObservations:
                idx = self.mObservations[pKF]

                if pKF.mvuRight[idx] >= 0:
                    self.nObs -= 2
                else:
                    self.nObs -= 1

                del self.mObservations[pKF]

                if self.mpRefKF == pKF and self.mObservations:
                    self.mpRefKF = next(iter(self.mObservations))

                if self.nObs <= 2:
                    bBad = True

        if bBad:
            self.set_bad_flag()

    def get_observations(self):
        with self.mMutexFeatures:
            return self.mObservations.copy()

    def observations(self):
        with self.mMutexFeatures:
            return self.nObs

    def set_bad_flag(self):
        with self.mMutexFeatures:
            with self.mMutexPos:

                self.mbBad = True
                obs = self.mObservations.copy()
                self.mObservations.clear()

        for pKF, idx in obs.items():
            pKF.erase_map_point_match_by_index(idx)

        self.mpMap.erase_map_point(self)

    def get_replaced(self):
        with self.mMutexFeatures:
            with self.mMutexPos:
                return self.mpReplaced

    def replace(self, pMP):
        if pMP.mnId == self.mnId:
            return

        with self.mMutexFeatures:
            with self.mMutexPos:
                obs = self.mObservations.copy()
                self.mObservations.clear()
                self.mbBad = True
                nvisible = self.mnVisible
                nfound = self.mnFound
                self.mpReplaced = pMP

        for pKF, idx in obs.items():
            if not pMP.is_in_key_frame(pKF):
                pKF.replace_map_point_match(idx, pMP)
                pMP.add_observation(pKF, idx)
            else:
                pKF.erase_map_point_match(idx)

        pMP.increase_found(nfound)
        pMP.increase_visible(nvisible)

        pMP.compute_distinctive_descriptors()

        self.mpMap.erase_map_point(self)

    def is_bad(self):
        with self.mMutexFeatures:
            with self.mMutexPos:
                return self.mbBad

    def increase_visible(self, n=1):
        with self.mMutexFeatures:
            self.mnVisible += n

    def increase_found(self, n=1):
        with self.mMutexFeatures:
            self.mnFound += n

    def get_found_ratio(self):
        with self.mMutexFeatures:
            if self.mnVisible > 0:
                return float(self.mnFound) / self.mnVisible
            else:
                return 0.0

    def compute_distinctive_descriptors(self):
        with self.mMutexFeatures:
            if self.mbBad:
                return
            observations = self.mObservations.copy()

        if not observations:
            return

        vDescriptors = []
        for pKF, idx in observations.items():
            if not pKF.is_bad():
                vDescriptors.append(pKF.mDescriptors[idx])

        if not vDescriptors:
            return

        N = len(vDescriptors)
        Distances = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i + 1, N):
                dist = self.descriptor_distance(vDescriptors[i], vDescriptors[j])
                Distances[i, j] = dist
                Distances[j, i] = dist

        BestMedian = float('inf')
        BestIdx = 0
        for i in range(N):
            dists = Distances[i]
            median = np.median(dists)
            if median < BestMedian:
                BestMedian = median
                BestIdx = i

        with self.mMutexFeatures:
            self.mDescriptor = vDescriptors[BestIdx].copy()

    def get_descriptor(self):
        with self.mMutexFeatures:
            return self.mDescriptor.copy()

    def get_index_in_key_frame(self, pKF):
        with self.mMutexFeatures:
            return self.mObservations.get(pKF, -1)

    def is_in_key_frame(self, pKF):
        with self.mMutexFeatures:
            return pKF in self.mObservations

    def update_normal_and_depth(self):
        with self.mMutexFeatures:
            with self.mMutexPos:
                if self.mbBad:
                    return
                observations = self.mObservations
                pRefKF = self.mpRefKF
                Pos = self.mWorldPos.copy()

        if not observations:
            return

        normal = np.zeros((3, 1), dtype=np.float32)
        n = 0
        for pKF, idx in observations.items():
            Owi = pKF.get_camera_center()
            normali = self.mWorldPos - Owi
            normal += normali / np.linalg.norm(normali)
            n += 1

        PC = Pos - pRefKF.get_camera_center()
        dist = np.linalg.norm(PC)
        level = pRefKF.mvKeysUn[observations[pRefKF]].octave
        level_scale_factor = pRefKF.mvScaleFactors[level]
        n_levels = pRefKF.mnScaleLevels

        with self.mMutexPos:

            self.mfMaxDistance = dist * level_scale_factor
            self.mfMinDistance = self.mfMaxDistance / pRefKF.mvScaleFactors[n_levels - 1]
            self.mNormalVector = normal / n

    def get_min_distance_invariance(self):
        with self.mMutexPos:
            return 0.8 * self.mfMinDistance

    def get_max_distance_invariance(self):
        with self.mMutexPos:
            return 1.2 * self.mfMaxDistance

    def predict_scale(self, current_dist, pKF):

        with self.mMutexPos:
            ratio = self.mfMaxDistance / current_dist

        nScale = int(np.ceil(np.log(ratio) / pKF.mfLogScaleFactor))
        nScale = max(0, min(nScale, pKF.mnScaleLevels - 1))

        return nScale


