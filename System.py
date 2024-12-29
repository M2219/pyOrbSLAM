import threading

import cv2
import yaml

from pyDBoW.TemplatedVocabulary import TemplatedVocabulary
from KeyFrameDatabase import KeyFrameDatabase
from Map import Map
from FrameDrawer import FrameDrawer
from MapDrawer import MapDrawer
from Tracking import Tracking
from LocalMapping import LocalMapping
from Viewer import Viewer

class System:
    def __init__(self, strVocFile, strSettingsFile, sensor, bUseViewer):

        self.mMutexMode = threading.Lock()
        self.mMutexState = threading.Lock()
        self.mMutexReset = threading.Lock()

        self.mSensor = sensor
        self.mpViewer = None
        self.mbReset = False
        self.mbActivateLocalizationMode = False
        self.mbDeactivateLocalizationMode = False
        self.bUseViewer = bUseViewer

        with open(strSettingsFile, 'r') as f:
            fsSettings = yaml.load(f, Loader=yaml.SafeLoader)

        print("Loading ORB Vocabulary. This could take a while...")

        self.mpVocabulary  = TemplatedVocabulary(k=5, L=3, weighting="TF_IDF", scoring="L1_NORM")
        self.mpVocabulary.load_from_text_file(strVocFile)

        if not self.mpVocabulary.load_from_text_file(strVocFile):
            print(f"Wrong path to vocabulary. Failed to open at: {strVocFile}")
            exit(-1)

        print("Vocabulary loaded!")

        self.mpKeyFrameDatabase = KeyFrameDatabase(self.mpVocabulary)

        self.mpMap = Map()

        self.mpFrameDrawer = FrameDrawer(self.mpMap)
        self.mpMapDrawer = MapDrawer(self.mpMap, fsSettings)

        self.mpLocalMapper = LocalMapping(self.mpMap)

        self.mpTracker = Tracking(self, self.mpVocabulary, self.mpFrameDrawer, self.mpMapDrawer,
                                  self.mpMap, self.mpKeyFrameDatabase, fsSettings, self.mSensor)


        self.mptLocalMapping_thread = threading.Thread(target=self.mpLocalMapper.run)
        self.mptLocalMapping_thread.start()

        #self.mpLoopCloser = LoopClosing(self.mpMap, self.mpKeyFrameDatabase, self.mpVocabulary, self.mSensor, ss)
        #self.mptLoopClosing_thread = threading.Thread(target=self.mpLoopCloser.run)
        #print(self.bUseViewer)

        #if self.bUseViewer:
        #    self.mpViewer = Viewer(self, self.mpFrameDrawer, self.mpMapDrawer, self.mpTracker, fsSettings)
        #    self.mptViewer_thread = threading.Thread(target=self.mpViewer.run)
        #    self.mptViewer_thread.start()




        #self.mpTracker.set_local_mapper(self.mpLocalMapper)
        #self.mpTracker.set_loop_closing(self.mpLoopCloser)

        #self.mpLocalMapper.set_tracker(self.mpTracker)
        #self.mpLocalMapper.set_loop_closer(self.mpLoopCloser)

        #self.mpLoopCloser.set_tracker(self.mpTracker)
        #self.mpLoopCloser.set_local_mapper(self.mpLocalMapper)

        #self.mpTracker.set_viewer(self.mpViewer)

    def reset(self):
        """
        Requests a system reset in a thread-safe manner.
        """
        with self.mMutexReset:
            self.mbReset = True

    def track_stereo(self, mleft, mright, timestamp):

        with self.mMutexMode:
            if self.mbActivateLocalizationMode:
                SLAM.mpLocalMapper.request_stop()

                # Wait until Local Mapping has effectively stopped
                while not self.mpLocalMapper.is_stopped():
                    time.sleep(0.001)

                self.mpTracker.inform_only_tracking(True)
                self.mbActivateLocalizationMode = False

            if self.mbDeactivateLocalizationMode:
                self.mpTracker.inform_only_tracking(False)
                self.mpLocalMapper.release()
                self.mbDeactivateLocalizationMode = False

        with self.mpLocalMapper.mMutexReset:  # Equivalent to after viewer and loop closing complete reset() function in the tracking
            if self.mbReset:
                self.mpTracker.reset()
                self.mbReset = False

        Tcw = self.mpTracker.grab_image_stereo(mleft, mright, timestamp)

        with self.mMutexState:
            self.mTrackingState = self.mpTracker.mState
            self.mTrackedMapPoints = self.mpTracker.mCurrentFrame.mvpMapPoints
            self.mTrackedKeyPointsUn = self.mpTracker.mCurrentFrame.mvKeysUn

        return Tcw


if __name__ == "__main__":

    strSettingsFile = "configs/KITTI00-02.yaml"
    strVocFile = "./Vocabulary/ORBvoc.txt"

    from stereo_kitti import LoadImages

    leftImages, rightImages, timeStamps = LoadImages("00")
    nImages = len(leftImages)

    SLAM = System(strVocFile, strSettingsFile, "Stereo", bUseViewer=True)

    for i in range(10):
        print("FFFFFFFFFFFFFFFFFFFrame = ", i)
        mleft = cv2.imread(leftImages[i], cv2.IMREAD_GRAYSCALE)
        mright = cv2.imread(rightImages[i], cv2.IMREAD_GRAYSCALE)
        timestamp = float(timeStamps[i])

        Tcw = SLAM.track_stereo(mleft, mright, timestamp)
        print(Tcw)

    SLAM.mptLocalMapping_thread.join()
    SLAM.mptViewer_thread.join()


