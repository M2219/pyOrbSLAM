import threading

import pypangolin as pangolin
import OpenGL.GL as gl
import numpy as np

import sys
import pydoc

def output_help_to_file(filepath, request):
    f = open(filepath, 'w')
    sys.stdout = f
    pydoc.help(request)
    f.close()
    sys.stdout = sys.__stdout__
    return


class MapDrawer:
    def __init__(self, pMap, fSettings):

        self.mMutexCamera = threading.Lock()

        self.mKeyFrameSize = fSettings["Viewer.KeyFrameSize"]
        self.mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"]
        self.mGraphLineWidth = fSettings["Viewer.GraphLineWidth"]
        self.mPointSize = fSettings["Viewer.PointSize"]
        self.mCameraSize = fSettings["Viewer.CameraSize"]
        self.mCameraLineWidth = fSettings["Viewer.CameraLineWidth"]
        self.mCameraPose = None
        self.mpMap = pMap

    def draw_map_points(self):
        # Retrieve map points and reference map points
        vpMPs = self.mpMap.get_all_map_points()

        #print("--->", len(vpMPs))

        vpRefMPs = self.mpMap.get_reference_map_points()
        spRefMPs = set(vpRefMPs)
        if not vpMPs:
            return

        # Draw non-reference map points
        gl.glPointSize(self.mPointSize)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(0.0, 0.0, 0.0)  # Black color

        for mp in vpMPs:
            if mp.is_bad() or mp in spRefMPs:
                continue
            pos = mp.get_world_pos()  # Assuming pos is a numpy array
            gl.glVertex3f(pos[0], pos[1], pos[2])

        gl.glEnd()


        # Draw reference map points
        gl.glPointSize(self.mPointSize)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(1.0, 0.0, 0.0)  # Red color

        for mp in spRefMPs:
            if mp.is_bad():
                continue
            pos = mp.get_world_pos()  # Assuming pos is a numpy array
            gl.glVertex3f(pos[0], pos[1], pos[2])

        gl.glEnd()



    def draw_key_frames(self, bDrawKF=True, bDrawGraph=True):
        w = self.mKeyFrameSize
        h = w * 0.75
        z = w * 0.6

        vpKFs = self.mpMap.get_all_key_frames()
        #print("--->", len(vpKFs))
        if bDrawKF:
            for pKF in vpKFs:
                Twc = pKF.get_pose_inverse().T  # Assuming Twc is a numpy array
                #print("TWC ---> ", Twc)
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
                vCovKFs = pKF.get_covisibles_by_weight(100)
                Ow = pKF.get_camera_center()  # Assuming Ow is a numpy array
                if vCovKFs:
                    for covKF in vCovKFs:
                        if covKF.mnId < pKF.mnId:
                            continue
                        Ow2 = covKF.get_camera_center()
                        gl.glVertex3f(Ow[0], Ow[1], Ow[2])
                        gl.glVertex3f(Ow2[0], Ow2[1], Ow2[2])

                # Spanning tree
                pParent = pKF.get_parent()
                if pParent:
                    Owp = pParent.get_camera_center()
                    gl.glVertex3f(Ow[0], Ow[1], Ow[2])
                    gl.glVertex3f(Owp[0], Owp[1], Owp[2])

                # Loops
                sLoopKFs = pKF.get_loop_edges()
                for loopKF in sLoopKFs:
                    if loopKF.mnId < pKF.mnId:
                        continue
                    Owl = loopKF.get_camera_center()
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
        #output_help_to_file(r'test.txt', Twc)

        gl.glMultMatrixf(Twc.Matrix().flatten())  # Assuming Twc.m is a flattened matrix compatible with OpenGL

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

    def get_current_OpenGL_camera_matrix(self):
        """
        Get the current camera pose in OpenGL format.

        Parameters:
        - M: pangolin.OpenGlMatrix instance to be updated with the camera pose.
        """

        #output_help_to_file(r'test.txt', M)

        m_Twc = np.eye(4, dtype=np.float32)

        if self.mCameraPose is not None:
            with self.mMutexCamera:
                # Extract rotation and translation
                Rwc = self.mCameraPose[:3, :3].T  # Transpose of the rotation matrix
                twc = -np.dot(Rwc, self.mCameraPose[:3, 3])  # Translation in world coordinates

            # Fill the OpenGL matrix
            m_Twc[0][0] = Rwc[0, 0]
            m_Twc[0][1] = Rwc[1, 0]
            m_Twc[0][2] = Rwc[2, 0]
            m_Twc[0][3] = 0.0

            m_Twc[1][0] = Rwc[0, 1]
            m_Twc[1][1] = Rwc[1, 1]
            m_Twc[1][2] = Rwc[2, 1]
            m_Twc[1][3] = 0.0

            m_Twc[2][0] = Rwc[0, 2]
            m_Twc[2][1] = Rwc[1, 2]
            m_Twc[2][2] = Rwc[2, 2]
            m_Twc[2][3] = 0.0

            m_Twc[3][0] = twc[0]
            m_Twc[3][1] = twc[1]
            m_Twc[3][2] = twc[2]
            m_Twc[3][3] = 1.0
            return m_Twc

        else:
            return np.eye(4, dtype=np.float32)

