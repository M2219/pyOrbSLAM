import threading
import cv2

class System:
    def __init__(self, strVocFile, strSettingsFile, sensor, bUseViewer):
        """
        Initializes the ORB-SLAM2 System.

        Args:
            strVocFile (str): Path to the vocabulary file.
            strSettingsFile (str): Path to the settings file.
            sensor (str): Type of input sensor ('MONOCULAR', 'STEREO', 'RGBD').
            bUseViewer (bool): Whether to use the viewer.
        """
        self.mSensor = sensor
        self.mpViewer = None
        self.mbReset = False
        self.mbActivateLocalizationMode = False
        self.mbDeactivateLocalizationMode = False

        # Welcome message
        print("\nORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza.")
        print("This program comes with ABSOLUTELY NO WARRANTY;")
        print("This is free software, and you are welcome to redistribute it")
        print("under certain conditions. See LICENSE.txt.\n")

        print("Input sensor was set to:", end=" ")
        if self.mSensor == "MONOCULAR":
            print("Monocular")
        elif self.mSensor == "STEREO":
            print("Stereo")
        elif self.mSensor == "RGBD":
            print("RGB-D")

        # Check settings file
        fsSettings = cv2.FileStorage(strSettingsFile, cv2.FILE_STORAGE_READ)
        if not fsSettings.isOpened():
            print(f"Failed to open settings file at: {strSettingsFile}")
            exit(-1)

        # Load ORB Vocabulary
        print("\nLoading ORB Vocabulary. This could take a while...")
        self.mpVocabulary = ORBVocabulary()
        if not self.mpVocabulary.load_from_text_file(strVocFile):
            print(f"Wrong path to vocabulary. Failed to open at: {strVocFile}")
            exit(-1)
        print("Vocabulary loaded!\n")

        # Create KeyFrame Database
        self.mpKeyFrameDatabase = KeyFrameDatabase(self.mpVocabulary)

        # Create the Map
        self.mpMap = Map()

        # Create Drawers for Viewer
        self.mpFrameDrawer = FrameDrawer(self.mpMap)
        self.mpMapDrawer = MapDrawer(self.mpMap, strSettingsFile)

        # Initialize the Tracking thread
        self.mpTracker = Tracking(self, self.mpVocabulary, self.mpFrameDrawer, self.mpMapDrawer,
                                  self.mpMap, self.mpKeyFrameDatabase, strSettingsFile, self.mSensor)

        # Initialize the Local Mapping thread
        self.mpLocalMapper = LocalMapping(self.mpMap, self.mSensor == "MONOCULAR")
        self.mptLocalMapping = threading.Thread(target=self.mpLocalMapper.run)

        # Initialize the Loop Closing thread
        self.mpLoopCloser = LoopClosing(self.mpMap, self.mpKeyFrameDatabase, self.mpVocabulary,
                                        self.mSensor != "MONOCULAR")
        self.mptLoopClosing = threading.Thread(target=self.mpLoopCloser.run)

        # Initialize the Viewer thread
        if bUseViewer:
            self.mpViewer = Viewer(self, self.mpFrameDrawer, self.mpMapDrawer, self.mpTracker, strSettingsFile)
            self.mptViewer = threading.Thread(target=self.mpViewer.run)
            self.mpTracker.set_viewer(self.mpViewer)

        # Set pointers between threads
        self.mpTracker.set_local_mapper(self.mpLocalMapper)
        self.mpTracker.set_loop_closing(self.mpLoopCloser)

        self.mpLocalMapper.set_tracker(self.mpTracker)
        self.mpLocalMapper.set_loop_closer(self.mpLoopCloser)

        self.mpLoopCloser.set_tracker(self.mpTracker)
        self.mpLoopCloser.set_local_mapper(self.mpLocalMapper)
