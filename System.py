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

class System:
    def __init__(self, strVocFile, strSettingsFile, sensor, bUseViewer):

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

        ss = {

             "pKF_ss": None, "pKF_ss_lock" : threading.lock,

             }


        self.mpTracker = Tracking(self, self.mpVocabulary, self.mpFrameDrawer, self.mpMapDrawer,
                                  self.mpMap, self.mpKeyFrameDatabase, fsSettings, self.mSensor, ss)

        self.mpLocalMapper = LocalMapping(self.mpMap, "MONOCULAR", ss=ss)
        self.mptLocalMapping_thread = threading.Thread(target=self.mpLocalMapper.run)

        #self.mpLoopCloser = LoopClosing(self.mpMap, self.mpKeyFrameDatabase, self.mpVocabulary, self.mSensor, ss)
        #self.mptLoopClosing_thread = threading.Thread(target=self.mpLoopCloser.run)

        #if self.bUseViewer:
        #    self.mpViewer = Viewer(self, self.mpFrameDrawer, self.mpMapDrawer, self.mpTracker, strSettingsFile, ss)
        #    self.mptViewer_thread = threading.Thread(target=self.mpViewer.run)



        #self.mpTracker.set_local_mapper(self.mpLocalMapper)
        #self.mpTracker.set_loop_closing(self.mpLoopCloser)

        #self.mpLocalMapper.set_tracker(self.mpTracker)
        #self.mpLocalMapper.set_loop_closer(self.mpLoopCloser)

        #self.mpLoopCloser.set_tracker(self.mpTracker)
        #self.mpLoopCloser.set_local_mapper(self.mpLocalMapper)

        #self.mpTracker.set_viewer(self.mpViewer)

