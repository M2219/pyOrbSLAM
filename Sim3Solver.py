import random
import math
import numpy as np
import cv2

class Sim3Solver:
    def __init__(self, pKF1, pKF2, vpMatched12, bFixScale):
        """
        Initializes the Sim3Solver.

        Args:
            pKF1 (KeyFrame): First KeyFrame.
            pKF2 (KeyFrame): Second KeyFrame.
            vpMatched12 (list): List of matched MapPoints between the two KeyFrames.
            bFixScale (bool): Whether to fix the scale.
        """
        self.mnIterations = 0
        self.mnBestInliers = 0
        self.mbFixScale = bFixScale

        self.mpKF1 = pKF1
        self.mpKF2 = pKF2

        vpKeyFrameMP1 = pKF1.get_map_point_matches()
        self.mN1 = len(vpMatched12)

        self.mvpMapPoints1 = []
        self.mvpMapPoints2 = []
        self.mvpMatches12 = vpMatched12
        self.mvnIndices1 = []
        self.mvX3Dc1 = []
        self.mvX3Dc2 = []
        self.mvnMaxError1 = []
        self.mvnMaxError2 = []

        Rcw1 = pKF1.get_rotation()
        tcw1 = pKF1.get_translation()
        Rcw2 = pKF2.get_rotation()
        tcw2 = pKF2.get_translation()

        self.mvAllIndices = []

        idx = 0
        for i1 in range(self.mN1):
            if vpMatched12[i1]:
                pMP1 = vpKeyFrameMP1[i1]
                pMP2 = vpMatched12[i1]

                if not pMP1 or pMP1.is_bad() or pMP2.is_bad():
                    continue

                indexKF1 = pMP1.get_index_in_key_frame(pKF1)
                indexKF2 = pMP2.get_index_in_key_frame(pKF2)

                if indexKF1 < 0 or indexKF2 < 0:
                    continue

                kp1 = pKF1.mvKeysUn[indexKF1]
                kp2 = pKF2.mvKeysUn[indexKF2]

                sigmaSquare1 = pKF1.mvLevelSigma2[kp1.octave]
                sigmaSquare2 = pKF2.mvLevelSigma2[kp2.octave]

                self.mvnMaxError1.append(9.210 * sigmaSquare1)
                self.mvnMaxError2.append(9.210 * sigmaSquare2)

                self.mvpMapPoints1.append(pMP1)
                self.mvpMapPoints2.append(pMP2)
                self.mvnIndices1.append(i1)

                X3D1w = pMP1.get_world_pos()
                self.mvX3Dc1.append(np.dot(Rcw1, X3D1w) + tcw1)

                X3D2w = pMP2.get_world_pos()
                self.mvX3Dc2.append(np.dot(Rcw2, X3D2w) + tcw2)

                self.mvAllIndices.append(idx)
                idx += 1

        self.mK1 = pKF1.mK
        self.mK2 = pKF2.mK

        self.mvP1im1 = self.from_camera_to_image(self.mvX3Dc1, self.mK1)
        self.mvP2im2 = self.from_camera_to_image(self.mvX3Dc2, self.mK2)

        self.set_ransac_parameters()

    def set_ransac_parameters(self, probability=0.99, minInliers=6 , maxIterations=300):
        """
        Set the RANSAC parameters and adjust iterations based on correspondences.
        """
        self.mRansacProb = probability
        self.mRansacMinInliers = minInliers
        self.mRansacMaxIts = maxIterations

        # Number of correspondences
        self.N = len(self.mvpMapPoints1)
        self.mvbInliersi = [False] * self.N

        # Adjust parameters based on number of correspondences
        epsilon = float(self.mRansacMinInliers) / self.N if self.N > 0 else 0.0

        # Compute RANSAC iterations based on probability, epsilon, and max iterations
        if self.mRansacMinInliers == self.N:
            n_iterations = 1
        else:
            n_iterations = math.ceil(
                math.log(1 - self.mRansacProb) / math.log(1 - math.pow(epsilon, 3))
            ) if epsilon > 0 else self.mRansacMaxIts

        self.mRansacMaxIts = max(1, min(n_iterations, self.mRansacMaxIts))
        self.mnIterations = 0

    def iterate(self, nIterations):
        """
        Perform RANSAC iterations to estimate the best Sim3 transformation.
        """
        vbInliers = [False] * self.mN1
        nInliers = 0
        bNoMore = False

        if self.N < self.mRansacMinInliers:
            bNoMore = True
            return None, bNoMore, vbInliers, nInliers

        P3Dc1i = np.zeros((3, 3), dtype=np.float32)
        P3Dc2i = np.zeros((3, 3), dtype=np.float32)

        nCurrentIterations = 0
        while self.mnIterations < self.mRansacMaxIts and nCurrentIterations < nIterations:
            nCurrentIterations += 1
            self.mnIterations += 1

            # Select 3 random points
            vAvailableIndices = self.mvAllIndices[:]
            for i in range(3):
                randi = random.randint(0, len(vAvailableIndices) - 1)
                idx = vAvailableIndices.pop(randi)

                P3Dc1i[:, i:i+1] = self.mvX3Dc1[idx]
                P3Dc2i[:, i:i+1] = self.mvX3Dc2[idx]

            # Compute Sim3
            self.compute_sim3(P3Dc1i, P3Dc2i)

            # Check inliers
            self.check_inliers()

            if self.mnInliersi >= self.mnBestInliers:
                self.mvbBestInliers = self.mvbInliersi
                self.mnBestInliers = self.mnInliersi
                self.mBestT12 = np.copy(self.mT12i)
                self.mBestRotation = np.copy(self.mR12i)
                self.mBestTranslation = np.copy(self.mt12i)
                self.mBestScale = self.ms12i

                if self.mnInliersi > self.mRansacMinInliers:
                    nInliers = self.mnInliersi
                    for i in range(self.N):
                        if self.mvbInliersi[i]:
                            vbInliers[self.mvAllIndices[i]] = True
                    return self.mBestT12, bNoMore, vbInliers, nInliers

        if self.mnIterations >= self.mRansacMaxIts:
            bNoMore = True

        return None, bNoMore, vbInliers, nInliers

    def find(self):
        """
        Wrapper for the `iterate` method that runs RANSAC iterations to find the best Sim3 transformation.
        """
        self.mBestT12, bNoMore, vbInliers12, nInliers = self.iterate(self.mRansacMaxIts)
        return vbInliers12, nInliers

    def compute_centroid(self, P):
        """
        Compute the centroid of a set of points and return the centered points.

        Parameters:
        - P: Input matrix of points (3 x N).
        - Pr: Output matrix of centered points (3 x N).
        """
        # Compute the centroid as the mean along the columns
        C = np.mean(P, axis=1, keepdims=True)

        # Center the points by subtracting the centroid
        Pr = P - C

        return Pr, C

    def compute_sim3(self, P1, P2):
        """
        Compute the Sim3 transformation between two sets of points P1 and P2.
        """
        # Step 1: Centroid and relative coordinates
        Pr1, O1 = self.compute_centroid(P1)
        Pr2, O2 = self.compute_centroid(P2)

        # Step 2: Compute M matrix
        M = Pr2 @ Pr1.T

        # Step 3: Compute N matrix
        N = np.zeros((4, 4), dtype=np.float32)
        N[0, 0] = M[0, 0] + M[1, 1] + M[2, 2]
        N[0, 1] = M[1, 2] - M[2, 1]
        N[0, 2] = M[2, 0] - M[0, 2]
        N[0, 3] = M[0, 1] - M[1, 0]
        N[1, 1] = M[0, 0] - M[1, 1] - M[2, 2]
        N[1, 2] = M[0, 1] + M[1, 0]
        N[1, 3] = M[2, 0] + M[0, 2]
        N[2, 2] = -M[0, 0] + M[1, 1] - M[2, 2]
        N[2, 3] = M[1, 2] + M[2, 1]
        N[3, 3] = -M[0, 0] - M[1, 1] + M[2, 2]

        # Fill symmetric elements
        N = N + N.T - np.diag(N.diagonal())

        # Step 4: Eigenvector of the highest eigenvalue
        evals, evecs = np.linalg.eig(N)
        max_eigen_idx = np.argmax(evals)
        quat = evecs[:, max_eigen_idx]  # Quaternion

        # Convert quaternion to angle-axis
        vec = quat[1:4]
        sin_angle = np.linalg.norm(vec)
        cos_angle = quat[0]
        ang = np.arctan2(sin_angle, cos_angle)
        vec = 2 * ang * vec / sin_angle if sin_angle != 0 else np.zeros_like(vec)

        # Compute rotation matrix from angle-axis representation
        self.mR12i = cv2.Rodrigues(vec)[0]

        # Step 5: Rotate set 2
        P3 = self.mR12i @ Pr2

        # Step 6: Scale
        if not self.mbFixScale:
            nom = np.sum(Pr1 * P3)
            den = np.sum(P3**2)
            self.ms12i = nom / den
        else:
            self.ms12i = 1.0

        # Step 7: Translation
        self.mt12i = O1 - self.ms12i * self.mR12i @ O2

        # Step 8: Transformation matrices
        self.mT12i = np.eye(4, dtype=np.float32)
        self.mT12i[:3, :3] = self.ms12i * self.mR12i
        self.mT12i[:3, 3] = self.mt12i.flatten()

        self.mT21i = np.eye(4, dtype=np.float32)
        self.mT21i[:3, :3] = (1.0 / self.ms12i) * self.mR12i.T
        self.mT21i[:3, 3:4] = -(1.0 / self.ms12i) * self.mR12i.T @ self.mt12i

    def check_inliers(self):
        """
        Check for inliers between sets of 3D points using the current Sim3 transformation.
        """

        vP2im1 = self.project(self.mvX3Dc2, self.mT12i, self.mK1)
        vP1im2 = self.project(self.mvX3Dc1, self.mT21i, self.mK2)

        self.mnInliersi = 0
        for i in range(len(self.mvP1im1)):
            dist1 = self.mvP1im1[i] - vP2im1[i]
            dist2 = vP1im2[i] - self.mvP2im2[i]

            err1 = np.dot(dist1.flatten(), dist1.flatten())
            err2 = np.dot(dist2.flatten(), dist2.flatten())

            if err1 < self.mvnMaxError1[i] and err2 < self.mvnMaxError2[i]:
                self.mvbInliersi[i] = True
                self.mnInliersi += 1
            else:
                self.mvbInliersi[i] = False

    def get_estimated_rotation(self):
        """
        Get the estimated rotation matrix from the best Sim3 transformation.
        """
        return self.mBestRotation.copy()

    def get_estimated_translation(self):
        """
        Get the estimated translation vector from the best Sim3 transformation.
        """
        return self.mBestTranslation.copy()

    def get_estimated_scale(self):
        """
        Get the estimated scale factor from the best Sim3 transformation.
        """
        return self.mBestScale

    def project(self, vP3Dw, Tcw, K):
        """
        Project 3D points in world coordinates into 2D image coordinates.

        Parameters:
        - vP3Dw: List of 3D points in world coordinates.
        - vP2D: List to store 2D image coordinates.
        - Tcw: Transformation matrix (camera to world).
        - K: Camera intrinsic matrix.
        """
        Rcw = Tcw[:3, :3]
        tcw = Tcw[:3, 3]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        vP2D = []

        for P3Dw in vP3Dw:
            P3Dc = Rcw @ P3Dw + tcw
            invz = 1.0 / P3Dc[2]
            x = P3Dc[0] * invz
            y = P3Dc[1] * invz
            vP2D.append(np.array([fx * x + cx, fy * y + cy], dtype=np.float32))

        return vP2D

    def from_camera_to_image(self, vP3Dc, K):
        """
        Project 3D points in camera coordinates into 2D image coordinates.

        Parameters:
        - vP3Dc: List of 3D points in camera coordinates.
        - vP2D: List to store 2D image coordinates.
        - K: Camera intrinsic matrix.
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        vP2D = []

        for P3Dc in vP3Dc:
            invz = 1.0 / P3Dc[2]
            x = P3Dc[0] * invz
            y = P3Dc[1] * invz
            vP2D.append(np.array([fx * x + cx, fy * y + cy], dtype=np.float32))

        return vP2D


