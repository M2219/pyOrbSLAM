import threading
import time

import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np
import cv2

class Viewer:
    def __init__(self, pSystem, pFrameDrawer, pMapDrawer, pTracker, fSettings):

        self.mutexFinish = threading.Lock()
        self.mutexStop = threading.Lock()

        self.mpSystem = pSystem
        self.mpFrameDrawer = pFrameDrawer
        self.mpMapDrawer = pMapDrawer
        self.mpTracker = pTracker

        self.mbFinishRequested = False
        self.mbFinished = True
        self.mbStopped = True
        self.mbStopRequested = False

        fps = fSettings["Camera.fps"]
        self.mT = 1e3 / (fps if fps >= 1 else 30)

        self.mImageWidth = int(fSettings["Camera.width"])
        self.mImageHeight = int(fSettings["Camera.height"])

        if self.mImageWidth < 1 or self.mImageHeight < 1:
            self.mImageWidth = 640
            self.mImageHeight = 480

        self.mViewpointX = fSettings["Viewer.ViewpointX"]
        self.mViewpointY = fSettings["Viewer.ViewpointY"]
        self.mViewpointZ = fSettings["Viewer.ViewpointZ"]
        self.mViewpointF = fSettings["Viewer.ViewpointF"]

    def run(self):

        print("viewer")

        self.mbFinished = False
        self.mbStopped = False

        pangolin.CreateWindowAndBind("pyOrbSLAM: Map Viewer", 1024, 768)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Define Camera Render Object
        s_cam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1024, 768, self.mViewpointF, self.mViewpointF, 512, 389, 0.1, 1000),
            pangolin.ModelViewLookAt(self.mViewpointX, self.mViewpointY, self.mViewpointZ, 0, 0, 0, 0, -1, 0)
        )

        # Create interactive panel
        #panel = pangolin.CreatePanel("menu").SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach(0.0), pangolin.Attach.Pix(175))
        menuFollowCamera = True
        menuShowPoints =  True
        menuShowKeyFrames =  True
        menuShowGraph =  True
        menuLocalizationMode =  False
        menuReset = False

        # Create Display
        d_cam = pangolin.CreateDisplay()
        d_cam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0), pangolin.Attach.Pix(175), pangolin.Attach(1.0), -1024.0 / 768.0)
        d_cam.SetHandler(pangolin.Handler3D(s_cam))


        # Control Flags
        bFollow = True
        bLocalizationMode = False

        while True:
            time.sleep(0.2)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            M = self.mpMapDrawer.get_current_OpenGL_camera_matrix()

            Twc = pangolin.OpenGlMatrix(M.T)

            if menuFollowCamera and bFollow:
                s_cam.Follow(Twc)

            elif menuFollowCamera and not bFollow:
                s_cam.SetModelViewMatrix(
                    pangolin.ModelViewLookAt(self.mViewpointX, self.mViewpointY, self.mViewpointZ, 0, 0, 0, 0, -1, 0)
                )

                s_cam.Follow(Twc)
                bFollow = True
            elif not menuFollowCamera and bFollow:
                bFollow = False

            if menuLocalizationMode and not bLocalizationMode:
                # Activate localization mode
                self.mpSystem.activate_localization_mode()
                bLocalizationMode = True

            elif not menuLocalizationMode and bLocalizationMode:
                # Deactivate localization mode
                self.mpSystem.deactivate_localization_mode()
                bLocalizationMode = False

            # Activate 3D camera view
            d_cam.Activate(s_cam)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)

            self.mpMapDrawer.draw_current_camera(Twc);

            if menuShowKeyFrames or menuShowGraph:
                self.mpMapDrawer.draw_key_frames(menuShowKeyFrames, menuShowGraph)

            if menuShowPoints:
                self.mpMapDrawer.draw_map_points()

            # Finish rendering frame
            pangolin.FinishFrame()

            # Show current frame (dummy implementation)
            img = self.mpFrameDrawer.draw_frame()
            cv2.imshow("pyOrbSLAM: Current Frame", img)
            cv2.waitKey(int(self.mT))

            if menuReset:
                menuShowGraph = True
                menuShowKeyFrames = True
                menuShowPoints = True
                menuLocalizationMode = False
                if bLocalizationMode:
                    self.mpSystem.deactivate_localization_mode()

                bLocalizationMode = False
                bFollow = True
                menuFollowCamera = True
                self.mpSystem.reset();
                menuReset = False

            # Add stop conditions (if needed)
            if self.stop():
                while self.is_stopped():
                    time.sleep(3000)

            if self.check_finish():
                break;

            # Exit loop condition (replace with actual check)
            if pangolin.ShouldQuit():
                break

        self.set_finish()

    def request_finish(self):
        with self.mutexFinish:
            self.mbFinishRequested = True

    def check_finish(self):
        with self.mutexFinish:
            return self.mbFinishRequested

    def set_finish(self):
        with self.mutexFinish:
            self.mbFinished = True

    def is_finished(self):
        with self.mutexFinish:
            return self.mbFinished

    def request_stop(self):
        with self.mutexStop:
            if not self.mbStopped:
                self.mbStopRequested = True

    def is_stopped(self):
        with self.mutexStop:
            return self.mbStopped

    def stop(self):
        with self.mutexStop, self.mutexFinish:
            if self.mbFinishRequested:
                return False
            elif self.mbStopRequested:
                self.mbStopped = True
                self.mbStopRequested = False
                return True
            return False

    def release(self):
        with self.mutexStop:
            self.mbStopped = False















