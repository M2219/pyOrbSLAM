import threading

from ordered_set import OrderedSet

class Map:
    def __init__(self):
        self.mMutexMapUpdate = threading.Lock()
        self.mMutexPointCreation = threading.Lock()
        self.mMutexMap = threading.Lock()

        self.mspKeyFrames = []
        self.mspMapPoints = []
        self.mvpReferenceMapPoints = []
        self.mvpKeyFrameOrigins = []
        self.mnMaxKFid = 0
        self.mnBigChangeIdx = 0

    def add_key_frame(self, pKF):
        with self.mMutexMap:
            self.mspKeyFrames.append(pKF)
            if pKF.mnId > self.mnMaxKFid:
                self.mnMaxKFid = pKF.mnId

    def add_map_point(self, pMP):

        with self.mMutexMap:
            self.mspMapPoints.append(pMP)

    def erase_map_point(self, pMP):
        with self.mMutexMap:
            if pMP in self.mspMapPoints:
                self.mspMapPoints.remove(pMP)

    def erase_key_frame(self, pKF):
        with self.mMutexMap:
            self.mspKeyFrames.remove(pKF)

    def set_reference_map_points(self, vpMPs):
        with self.mMutexMap:
            self.mvpReferenceMapPoints = vpMPs

    def inform_new_big_change(self):
        with self.mMutexMap:
            self.mnBigChangeIdx += 1

    def get_last_big_change_idx(self):
        with self.mMutexMap:
            return self.mnBigChangeIdx

    def get_all_key_frames(self):
        with self.mMutexMap:
            return OrderedSet(self.mspKeyFrames)

    def get_all_map_points(self):
        with self.mMutexMap:
            return OrderedSet(self.mspMapPoints)

    def map_points_in_map(self):
        with self.mMutexMap:
            return len(self.mspMapPoints)

    def key_frames_in_map(self):
        with self.mMutexMap:
            return len(self.mspKeyFrames)

    def get_reference_map_points(self):
        with self.mMutexMap:
            return self.mvpReferenceMapPoints

    def get_max_kf_id(self):
        with self.mMutexMap:
            return self.mnMaxKFid

    def clear(self):
        for pMP in list(self.mspMapPoints):
            self.mspMapPoints.remove(pMP)

        for pKF in list(self.mspKeyFrames):
            self.mspKeyFrames.remove(pKF)

        self.mspMapPoints.clear()
        self.mspKeyFrames.clear()
        self.mnMaxKFid = 0
        self.mvpReferenceMapPoints.clear()
        self.mvpKeyFrameOrigins.clear()


