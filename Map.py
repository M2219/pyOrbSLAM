class Map:
    def __init__(self):
        self.mspKeyFrames = set()  # Set of KeyFrames
        self.mspMapPoints = set()  # Set of MapPoints
        self.mvpReferenceMapPoints = []  # List of reference MapPoints
        self.mvpKeyFrameOrigins = []  # List of KeyFrame origins
        self.mnMaxKFid = 0  # Maximum KeyFrame ID
        self.mnBigChangeIdx = 0  # Index for tracking big changes

    def add_key_frame(self, pKF):
        """Adds a KeyFrame to the map."""
        self.mspKeyFrames.add(pKF)
        if pKF.mnId > self.mnMaxKFid:
            self.mnMaxKFid = pKF.mnId

    def add_map_point(self, pMP):
        """Adds a MapPoint to the map."""
        self.mspMapPoints.add(pMP)

    def erase_map_point(self, pMP):
        """Erases a MapPoint from the map."""
        with self.mMutexMap:
            self.mspMapPoints.discard(pMP)
            # TODO: This only removes the reference. Actual deletion logic may be required.

    def erase_key_frame(self, pKF):
        """Erases a KeyFrame from the map."""
        with self.mMutexMap:
            self.mspKeyFrames.discard(pKF)
            # TODO: This only removes the reference. Actual deletion logic may be required.

    def set_reference_map_points(self, vpMPs):
        """Sets the reference MapPoints."""
        with self.mMutexMap:
            self.mvpReferenceMapPoints = vpMPs

    def inform_new_big_change(self):
        """Increments the big change index."""
        with self.mMutexMap:
            self.mnBigChangeIdx += 1

    def get_last_big_change_idx(self):
        """Retrieves the index of the last big change."""
        with self.mMutexMap:
            return self.mnBigChangeIdx

    def get_all_key_frames(self):
        """Retrieves all KeyFrames."""
        with self.mMutexMap:
            return list(self.mspKeyFrames)

    def get_all_map_points(self):
        """Retrieves all MapPoints."""
        with self.mMutexMap:
            return list(self.mspMapPoints)

    def map_points_in_map(self):
        """Retrieves the count of MapPoints in the map."""
        with self.mMutexMap:
            return len(self.mspMapPoints)

    def key_frames_in_map(self):
        """Retrieves the count of KeyFrames in the map."""
        with self.mMutexMap:
            return len(self.mspKeyFrames)

    def get_reference_map_points(self):
        """Retrieves the reference MapPoints."""
        with self.mMutexMap:
            return self.mvpReferenceMapPoints

    def get_max_kf_id(self):
        """Retrieves the maximum KeyFrame ID."""
        with self.mMutexMap:
            return self.mnMaxKFid

    def clear(self):
        """Clears the map, deleting all KeyFrames and MapPoints."""
        with self.mMutexMap:
            for pMP in list(self.mspMapPoints):
                del pMP  # Assuming explicit deletion is required

            for pKF in list(self.mspKeyFrames):
                del pKF  # Assuming explicit deletion is required

            self.mspMapPoints.clear()
            self.mspKeyFrames.clear()
            self.mnMaxKFid = 0
            self.mvpReferenceMapPoints.clear()
            self.mvpKeyFrameOrigins.clear()


if __name__ == "__main__":



    mMap = Map()
