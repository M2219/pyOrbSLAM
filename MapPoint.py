import threading
import numpy as np

from Map import ss_map

class MapPoint:
    nNextId = 0  # Class-level variable for unique MapPoint IDs

    def __init__(self, Pos, pRefKF, pMap):
        """
        Initializes a MapPoint.

        Args:
            Pos (np.ndarray): 3D position in the world (3x1 matrix).
            pRefKF (KeyFrame): Reference KeyFrame.
            pMap (Map): The map to which this MapPoint belongs.
        """
        self.mGlobalMutex = threading.Lock()
        self.mMutexPos = threading.Lock()
        self.mMutexFeatures = threading.Lock()

        self.mWorldPos = Pos.copy()  # Copy the position
        self.mpRefKF = pRefKF  # Reference KeyFrame
        self.mpMap = pMap  # Map reference

        self.mnFirstKFid = pRefKF.mnId  # ID of the first KeyFrame observing this MapPoint
        self.mnFirstFrame = pRefKF.mnFrameId  # ID of the first frame observing this MapPoint

        self.nObs = 0  # Number of observations
        self.mnTrackReferenceForFrame = 0  # Tracking reference for the current frame
        self.mnLastFrameSeen = 0  # Last frame where this MapPoint was seen

        self.mnBALocalForKF = 0  # Local bundle adjustment KeyFrame
        self.mnFuseCandidateForKF = 0  # Fuse candidate KeyFrame
        self.mnLoopPointForKF = 0  # Loop KeyFrame
        self.mnCorrectedByKF = 0  # Corrected by KeyFrame
        self.mnCorrectedReference = 0  # Corrected reference
        self.mnBAGlobalForKF = 0  # Global bundle adjustment KeyFrame

        self.mnVisible = 1  # Number of times this MapPoint was visible
        self.mnFound = 1  # Number of times this MapPoint was found

        self.mbBad = False  # Flag indicating if this MapPoint is bad
        self.mpReplaced = None  # Pointer to replaced MapPoint
        self.mObservations = {}  # Pointer to replaced MapPoint

        self.mfMinDistance = 0  # Minimum distance to the camera
        self.mfMaxDistance = 0  # Maximum distance to the camera

        self.mNormalVector = np.zeros((3, 1), dtype=np.float32)  # Normal vector

        # Assign a unique ID with thread safety# Important --------------------------------------------------------
        self.mnId = MapPoint.nNextId
        with mpMap.mMutexPointCreation:
            MapPoint.nNextId += 1

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

    def get_observations():
        with self.mMutexFeatures:
            return self.mObservations.copy()

    def observation():
        with self.mMutexFeatures:
            return self.nObs


    def set_bad_flag(self):

        with self.mMutexFeatures:
            with self.mMutexPos:

                self.mbBad = True
                obs = self.mObservations.copy()
                self.mObservations.clear()

        for pKF, idx in obs.items():
            pKF.erase_map_point_match(idx)

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

        # Update KeyFrames to replace this MapPoint with the new one
        for pKF, idx in obs.items():
            if not pMP.is_in_key_frame(pKF):
                pKF.replace_map_point_match(idx, pMP)
                pMP.add_observation(pKF, idx)
            else:
                pKF.erase_map_point_match(idx)

        # Update the new MapPoint's visibility and found counters
        pMP.increase_found(nfound)
        pMP.increase_visible(nvisible)

        # Recompute distinctive descriptors for the new MapPoint
        pMP.compute_distinctive_descriptors()

        # Remove this MapPoint from the map
        self.mpMap.erase_map_point(self)


    def is_bad(self):
        with self.mMutexFeatures:
            with self.mMutexPos:
                return self.mbBad

    def increase_visible(self, n):
        with self.mMutexFeatures:
            self.mnVisible += n

    def increase_found(self, n):
        with self.mMutexFeatures:
            self.mnFound += n

    def get_found_ratio(self):
        with self.mMutexFeatures:
            if self.mnVisible > 0:
                return float(self.mnFound) / self.mnVisible
            else:
                return 0.0

    def compute_distinctive_descriptors(self):
        """
        Computes the distinctive descriptor for the MapPoint based on the median distance
        to other descriptors.
        """
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
                dist = ORBmatcher.descriptor_distance(vDescriptors[i], vDescriptors[j])
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
            ss_mappoint["mDescriptor"] = vDescriptors[BestIdx]

    def get_descriptor(self):
        with self.mMutexFeatures:
            return ss_mappoint["mDescriptor"].copy() if ss_mappoint["mDescriptor"] is not None else None

    def get_index_in_key_frame(self, pKF):
        with self.mMutexFeatures:
            if (len(self.mObservations) > 0):
                return self.mObservations[pKF], -1)
            else
                return -1

    def is_in_key_frame(self, pKF):
        with self.mMutexFeatures:
            return pKF in self.mObservations

    def update_normal_and_depth(self):
        """
        Updates the normal vector and depth range of the MapPoint based on its observations.
        """
        with self.mMutexFeatures:
            with self.mMutexPos:
                if self.mbBad:
                    return
                observations = self.mObservations.copy()
                pRefKF = self.mpRefKF
                Pos = self.mWorldPos.copy()

        if not observations:
            return

        normal = np.zeros((3, 1), dtype=np.float32)
        n = 0
        for pKF, idx in observations.items():
            Owi = pKF.get_camera_center()
            normali = Pos - Owi
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

    def predict_scale(self, current_dist, pKF): # there  is another predict_scale which takes the frame

        with self.mMutexPos:
            ratio = self.mfMaxDistance / current_dist

        nScale = int(np.ceil(np.log(ratio) / pKF.mfLogScaleFactor))
        nScale = max(0, min(nScale, pKF.mnScaleLevels - 1))

        return nScale



if __name__ == "__main__":

    m = MapPoint(None, None, None)
