import threading

import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np

class MapDrawer:
    def __init__(self, pMap, fSettings):

        self.mMutexCamera = threading.Lock()

        self.mKeyFrameSize = fSettings["Viewer.KeyFrameSize"]
        self.mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"]
        self.mGraphLineWidth = fSettings["Viewer.GraphLineWidth"]
        self.mPointSize = fSettings["Viewer.PointSize"]
        self.mCameraSize = fSettings["Viewer.CameraSize"]
        self.mCameraLineWidth = fSettings["Viewer.CameraLineWidth"]

        self.mKeyFrameSize = 1.0
        self.mKeyFrameLineWidth = 1.0
        self.mGraphLineWidth = 1.0
        self.mCameraSize = 1.0
        self.mCameraLineWidth = 1.0
        self.mCameraPose = None

    def draw_map_points(self):
        # Retrieve map points and reference map points
        vpMPs = self.mpMap.GetAllMapPoints()
        vpRefMPs = self.mpMap.GetReferenceMapPoints()

        spRefMPs = set(vpRefMPs)

        if not vpMPs:
            return

        # Draw non-reference map points
        gl.glPointSize(self.mPointSize)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(0.0, 0.0, 0.0)  # Black color

        for mp in vpMPs:
            if mp.isBad() or mp in spRefMPs:
                continue
            pos = mp.GetWorldPos()  # Assuming pos is a numpy array
            gl.glVertex3f(pos[0], pos[1], pos[2])

        gl.glEnd()

        # Draw reference map points
        gl.glPointSize(self.mPointSize)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(1.0, 0.0, 0.0)  # Red color

        for mp in spRefMPs:
            if mp.isBad():
                continue
            pos = mp.GetWorldPos()  # Assuming pos is a numpy array
            gl.glVertex3f(pos[0], pos[1], pos[2])

        gl.glEnd()



    def draw_key_frames(self, bDrawKF=True, bDrawGraph=True):
        w = self.mKeyFrameSize
        h = w * 0.75
        z = w * 0.6

        vpKFs = self.mpMap.GetAllKeyFrames()

        if bDrawKF:
            for pKF in vpKFs:
                Twc = pKF.GetPoseInverse().T  # Assuming Twc is a numpy array
                gl.glPushMatrix()
                gl.glMultMatrixf(Twc.flatten())  # Flatten for OpenGL compatibility
                gl.glLineWidth(self.mKeyFrameLineWidth)
                gl.glColor3f(0.0, 0.0, 1.0)  # Blue color
                gl.glBegin(gl.GL_LINES)
                # Draw the pyramid representing the keyframe
                gl.glVertex3f(0, 0, 0)
                gl.glVertex3f(w, h, z)
                gl.glVertex3f(0, 0, 0)
                gl.glVertex3f(w, -h, z)
                gl.glVertex3f(0, 0, 0)
                gl.glVertex3f(-w, -h, z)
                gl.glVertex3f(0, 0, 0)
                gl.glVertex3f(-w, h, z)

                gl.glVertex3f(w, h, z)
                gl.glVertex3f(w, -h, z)
                gl.glVertex3f(-w, h, z)
                gl.glVertex3f(-w, -h, z)
                gl.glVertex3f(-w, h, z)
                gl.glVertex3f(w, h, z)
                gl.glVertex3f(-w, -h, z)
                gl.glVertex3f(w, -h, z)
                gl.glEnd()
                gl.glPopMatrix()

        if bDrawGraph:
            gl.glLineWidth(self.mGraphLineWidth)
            gl.glColor4f(0.0, 1.0, 0.0, 0.6)  # Green color with transparency
            gl.glBegin(gl.GL_LINES)

            for pKF in vpKFs:
                # Covisibility Graph
                vCovKFs = pKF.GetCovisiblesByWeight(100)
                Ow = pKF.GetCameraCenter()  # Assuming Ow is a numpy array
                if vCovKFs:
                    for covKF in vCovKFs:
                        if covKF.mnId < pKF.mnId:
                            continue
                        Ow2 = covKF.GetCameraCenter()
                        gl.glVertex3f(Ow[0], Ow[1], Ow[2])
                        gl.glVertex3f(Ow2[0], Ow2[1], Ow2[2])

                # Spanning tree
                pParent = pKF.GetParent()
                if pParent:
                    Owp = pParent.GetCameraCenter()
                    gl.glVertex3f(Ow[0], Ow[1], Ow[2])
                    gl.glVertex3f(Owp[0], Owp[1], Owp[2])

                # Loops
                sLoopKFs = pKF.GetLoopEdges()
                for loopKF in sLoopKFs:
                    if loopKF.mnId < pKF.mnId:
                        continue
                    Owl = loopKF.GetCameraCenter()
                    gl.glVertex3f(Ow[0], Ow[1], Ow[2])
                    gl.glVertex3f(Owl[0], Owl[1], Owl[2])

            gl.glEnd()

    def draw_current_camera(self, Twc):
        """
        Draw the current camera in the scene.
        Parameters:
        - Twc: Pangolin OpenGlMatrix (numpy array or similar structure)
        """
        w = self.mCameraSize
        h = w * 0.75
        z = w * 0.6

        gl.glPushMatrix()

        # Apply the camera transformation matrix
        gl.glMultMatrixd(Twc.m)  # Assuming Twc.m is a flattened matrix compatible with OpenGL

        gl.glLineWidth(self.mCameraLineWidth)
        gl.glColor3f(0.0, 1.0, 0.0)  # Green color for the camera
        gl.glBegin(gl.GL_LINES)

        # Draw the pyramid representing the camera
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(w, h, z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(w, -h, z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(-w, -h, z)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(-w, h, z)

        gl.glVertex3f(w, h, z)
        gl.glVertex3f(w, -h, z)

        gl.glVertex3f(-w, h, z)
        gl.glVertex3f(-w, -h, z)

        gl.glVertex3f(-w, h, z)
        gl.glVertex3f(w, h, z)

        gl.glVertex3f(-w, -h, z)
        gl.glVertex3f(w, -h, z)

        gl.glEnd()

        gl.glPopMatrix()

    def set_current_camera_pose(self, Tcw):
        """
        Set the current camera pose.
        Parameters:
        - Tcw: 4x4 numpy array representing the camera pose.
        """
        with self.mMutexCamera:
            self.mCameraPose = np.copy(Tcw)  # Clone the matrix

    def get_current_OpenGL_camera_matrix(self, M):
        """
        Get the current camera pose in OpenGL format.

        Parameters:
        - M: pangolin.OpenGlMatrix instance to be updated with the camera pose.
        """
        if self.mCameraPose is not None:
            with self.mMutexCamera:
                # Extract rotation and translation
                Rwc = self.mCameraPose[:3, :3].T  # Transpose of the rotation matrix
                twc = -np.dot(Rwc, self.mCameraPose[:3, 3])  # Translation in world coordinates

            # Fill the OpenGL matrix
            M.m[0] = Rwc[0, 0]
            M.m[1] = Rwc[1, 0]
            M.m[2] = Rwc[2, 0]
            M.m[3] = 0.0

            M.m[4] = Rwc[0, 1]
            M.m[5] = Rwc[1, 1]
            M.m[6] = Rwc[2, 1]
            M.m[7] = 0.0

            M.m[8] = Rwc[0, 2]
            M.m[9] = Rwc[1, 2]
            M.m[10] = Rwc[2, 2]
            M.m[11] = 0.0

            M.m[12] = twc[0]
            M.m[13] = twc[1]
            M.m[14] = twc[2]
            M.m[15] = 1.0
        else:
            M.SetIdentity()  # Set to identity if no pose is set


