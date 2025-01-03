import threading
import time
import cv2
import yaml

import pypangolin as pangolin
import numpy as np

from pyDBoW.TemplatedVocabulary import TemplatedVocabulary
from KeyFrameDatabase import KeyFrameDatabase
from Map import Map
from FrameDrawer import FrameDrawer
from MapDrawer import MapDrawer
from Tracking import Tracking
from LocalMapping import LocalMapping
from LoopClosing import LoopClosing
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

        self.mpKeyFrameDatabase = KeyFrameDatabase(self.mpVocabulary)

        self.mpMap = Map()

        self.mpFrameDrawer = FrameDrawer(self.mpMap)

        self.mpMapDrawer = MapDrawer(self.mpMap, fsSettings)


        self.mpTracker = Tracking(self, self.mpVocabulary, self.mpFrameDrawer, self.mpMapDrawer,
                                  self.mpMap, self.mpKeyFrameDatabase, fsSettings, self.mSensor)


        self.mpLocalMapper = LocalMapping(self, self.mpMap)
        self.mptLocalMapping_thread = threading.Thread(target=self.mpLocalMapper.run)
        self.mptLocalMapping_thread.start()

        self.mpLoopCloser = LoopClosing(self, self.mpMap, self.mpKeyFrameDatabase, self.mpVocabulary)
        self.mptLoopClosing_thread = threading.Thread(target=self.mpLoopCloser.run)
        self.mptLoopClosing_thread.start()

        if self.bUseViewer:
            self.mpViewer = Viewer(self, self.mpFrameDrawer, self.mpMapDrawer, self.mpTracker, fsSettings)
            self.mptViewer_thread = threading.Thread(target=self.mpViewer.run)
            self.mptViewer_thread.start()

    def reset(self):
        with self.mMutexReset:
            self.mbReset = True

    def track_stereo(self, mleft, mright, timestamp, i):

        with self.mMutexMode:
            if self.mbActivateLocalizationMode:
                SLAM.mpLocalMapper.request_stop()

                while not self.mpLocalMapper.is_stopped():
                    time.sleep(0.001)

                self.mpTracker.inform_only_tracking(True)
                self.mbActivateLocalizationMode = False

            if self.mbDeactivateLocalizationMode:
                self.mpTracker.inform_only_tracking(False)
                self.mpLocalMapper.release()
                self.mbDeactivateLocalizationMode = False

        with self.mpLocalMapper.mMutexReset:
            if self.mbReset:
                self.mpTracker.reset()
                self.mbReset = False

        Tcw = self.mpTracker.grab_image_stereo(mleft, mright, timestamp, i)

        with self.mMutexState:
            self.mTrackingState = self.mpTracker.mState
            self.mTrackedMapPoints = self.mpTracker.mCurrentFrame.mvpMapPoints
            self.mTrackedKeyPointsUn = self.mpTracker.mCurrentFrame.mvKeysUn

        return Tcw

    def activate_localization_mode(self):
        with self.mMutexMode:
            self.mbActivateLocalizationMode = True

    def deactivate_localization_mode(self):
        with self.mMutexMode:
            self.mbDeactivateLocalizationMode = True

    def save_trajectory_kitti(self, filename):

        print(f"Saving camera trajectory to {filename} ...")

        vpKFs = self.mpMap.get_all_key_frames()
        vpKFs_s = sorted(vpKFs, key=lambda kf: kf.mnId)

        Two = vpKFs_s[0].get_pose_inverse()

        with open(filename, "w") as f:
            for lit, lRit, lT in zip(self.mpTracker.mlRelativeFramePoses,
                                     self.mpTracker.mlpReferences,
                                     self.mpTracker.mlFrameTimes):
                pKF = lRit

                Trw = np.eye(4, dtype=np.float32)

                while pKF.is_bad():
                    Trw = np.dot(Trw, pKF.mTcp)
                    pKF = pKF.get_parent()

                Trw = np.dot(np.dot(Trw, pKF.get_pose()), Two)

                Tcw = np.dot(lit, Trw)
                Rwc = Tcw[:3, :3].T
                twc = -np.dot(Rwc, Tcw[:3, 3])

                f.write("{:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
                    Rwc[0, 0], Rwc[0, 1], Rwc[0, 2], twc[0],
                    Rwc[1, 0], Rwc[1, 1], Rwc[1, 2], twc[1],
                    Rwc[2, 0], Rwc[2, 1], Rwc[2, 2], twc[2]
                ))

        print("Trajectory saved!")

    def shutdown(self):
        self.mpLocalMapper.request_finish()
        self.mpLoopCloser.request_finish()

        if self.mpViewer:
            self.mpViewer.request_finish()
            while not self.mpViewer.is_finished():
                time.sleep(0.005)

        while (not self.mpLocalMapper.is_finished()):
            time.sleep(0.005)

        if self.mpViewer:
            self.mptViewer_thread.join()

        self.mptLocalMapping_thread.join()
        self.mpLoopCloser_thread.join()



