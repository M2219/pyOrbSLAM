import threading
import math
import time

import numpy as np
import cv2

from ORBMatcher import TH_HIGH, TH_LOW, HISTO_LENGTH

class Frame:

    nNextId = 0
    def __init__(self, mleft, mright, timestamp, mpORBextractorLeft, mpORBextractorRight,
                   mpVocabulary, mK, mDistCoef, mbf, mThDepth, frame_args):


        # self.args = args
        self.frame_args = frame_args
        self.fx = frame_args[0]
        self.fy = frame_args[1]
        self.cx = frame_args[2]
        self.cy = frame_args[3]
        self.invfx = frame_args[4]
        self.invfy = frame_args[5]
        self.mfGridElementWidthInv = frame_args[6]
        self.mfGridElementHeightInv = frame_args[7]
        self.mnMinX = frame_args[8]
        self.mnMaxX = frame_args[9]
        self.mnMinY = frame_args[10]
        self.mnMaxY = frame_args[11]
        self.FRAME_GRID_ROWS = frame_args[12]
        self.FRAME_GRID_COLS = frame_args[13]
        #
        self.mpORBvocabulary = mpVocabulary
        self.mbf = mbf
        self.mK =mK
        self.mDistCoef = mDistCoef
        self.mleft = mleft
        self.mright = mright
        self.mTimeStamp = timestamp
        self.mThDepth = mThDepth
        self.mBowVec = None
        self.mFeatVec = None

        self.mpORBextractorLeft = mpORBextractorLeft
        self.mpORBextractorRight = mpORBextractorRight

        self.mpReferenceKF = None

        self.mb = self.mbf / self.mK[0][0]
        #threadLeft = threading.Thread(target=self.ExtractORB, args=(0, mleft))
        #threadRight = threading.Thread(target=self.ExtractORB, args=(1, mright))

        #threadLeft.start()
        #threadRight.start()

        #threadLeft.join()
        #threadRight.join()

        self.ExtractORB(0, mleft)
        self.ExtractORB(1, mright)

        self.mnScaleLevels = mpORBextractorLeft.GetLevels()
        self.mfScaleFactor = mpORBextractorLeft.GetScaleFactor()
        self.mfLogScaleFactor = np.log(self.mfScaleFactor)
        self.mvScaleFactors = mpORBextractorLeft.GetScaleFactors()
        self.mvInvScaleFactors = mpORBextractorLeft.GetInverseScaleFactors()
        self.mvLevelSigma2 = mpORBextractorLeft.GetScaleSigmaSquares()
        self.mvInvLevelSigma2 = mpORBextractorLeft.GetInverseScaleSigmaSquares()

        self.mvImagePyramidLeft = mpORBextractorLeft.GetImagePyramid()
        self.mvImagePyramidRight = mpORBextractorRight.GetImagePyramid()

        self.N = len(self.mvKeys)

        self.undistort_keypoints()
        tt = time.time()
        self.compute_stereo_matches()
        print(f"------> time stereo matching : {time.time()-tt}")

        self.mvpMapPoints = [None] * self.N
        self.mvbOutlier = [False] * self.N

        #tt = time.time()

        self.assign_features_to_grid()

        self.mnId = Frame.nNextId
        Frame.nNextId += 1

        #print("stereo", time.time() - tt)

    def copy(self, frame):
        new_frame = Frame(self.mleft, self.mright, self.mTimeStamp, self.mpORBextractorLeft, self.mpORBextractorRight,
                           self.mpORBvocabulary, self.mK, self.mDistCoef, self.mbf, self.mThDepth, self.frame_args)
        new_frame.mpORBvocabulary = frame.mpORBvocabulary
        new_frame.mpORBextractorLeft = frame.mpORBextractorLeft
        new_frame.mpORBextractorRight = frame.mpORBextractorRight
        new_frame.mTimeStamp = frame.mTimeStamp
        new_frame.mK = frame.mK.copy()
        new_frame.mDistCoef = frame.mDistCoef.copy()
        new_frame.mbf = frame.mbf
        new_frame.mThDepth = frame.mThDepth
        new_frame.N = frame.N
        new_frame.mvKeys = frame.mvKeys
        new_frame.mvKeysRight = frame.mvKeysRight
        new_frame.mvKeysUn = frame.mvKeysUn
        new_frame.mvuRight = frame.mvuRight
        new_frame.mvDepth = frame.mvDepth
        new_frame.mBowVec = frame.mBowVec
        new_frame.mFeatVec = frame.mFeatVec
        new_frame.mDescriptors = frame.mDescriptors.copy()
        new_frame.mDescriptorsRight = frame.mDescriptorsRight.copy()
        new_frame.mvpMapPoints = frame.mvpMapPoints
        new_frame.mvbOutlier = frame.mvbOutlier
        new_frame.mnId = frame.mnId
        new_frame.mpReferenceKF = frame.mpReferenceKF
        new_frame.mnScaleLevels = frame.mnScaleLevels
        new_frame.mfScaleFactor = frame.mfScaleFactor
        new_frame.mfLogScaleFactor = frame.mfLogScaleFactor
        new_frame.mvScaleFactors = frame.mvScaleFactors
        new_frame.mvInvScaleFactors = frame.mvInvScaleFactors
        new_frame.mvLevelSigma2 = frame.mvLevelSigma2
        new_frame.mvInvLevelSigma2 = frame.mvInvLevelSigma2
        new_frame.mGrid = frame.mGrid

        if frame.mTcw is not None:
            new_frame.set_pose(frame.mTcw)

        return new_frame

    def ExtractORB(self, flag, image):
        if flag == 0:
            self.mvKeys_, self.mDescriptors = self.mpORBextractorLeft.operator_kd(image)
            self.mvKeys = [cv2.KeyPoint(*kp) for kp in self.mvKeys_]

        elif flag == 1:
            self.mvKeysRight_, self.mDescriptorsRight = self.mpORBextractorRight.operator_kd(image)
            self.mvKeysRight = [cv2.KeyPoint(*kp) for kp in self.mvKeysRight_]

    def compute_BoW(self):

         self.mBowVec, self.mFeatVec = self.mpORBvocabulary.transform(self.mDescriptors, 4)

    def set_pose(self, Tcw_):
        self.mTcw = Tcw_.copy()
        self.update_pose_matrices()

    def update_pose_matrices(self):
        self.mRcw = self.mTcw[:3, :3]
        self.mRwc = self.mRcw.T
        self.mtcw = self.mTcw[:3, 3].reshape(3, 1)
        self.mOw = -np.dot(self.mRwc, self.mtcw)

    def get_camera_center(self):
        return self.mOw.copy()

    def get_rotation_inverse(self):
        return self.mRwc.copy()

    def pos_in_grid(self, kps):
        # Vectorized computation for all keypoints
        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps])
        posX = np.round((keypoints[:, 0] - self.mnMinX) * self.mfGridElementWidthInv).astype(int)
        posY = np.round((keypoints[:, 1] - self.mnMinY) * self.mfGridElementHeightInv).astype(int)

        # Check bounds
        valid = (posX >= 0) & (posX < self.FRAME_GRID_COLS) & (posY >= 0) & (posY < self.FRAME_GRID_ROWS)

        return valid, posX, posY

    def assign_features_to_grid(self):
        # Initialize the grid with empty lists
        self.mGrid = [[[] for _ in range(self.FRAME_GRID_ROWS)] for _ in range(self.FRAME_GRID_COLS)]

        # Precompute grid positions for all keypoints
        valid, posX, posY = self.pos_in_grid(self.mvKeys)

        # Assign keypoints to grid cells
        for i in range(self.N):
            if valid[i]:
                self.mGrid[posX[i]][posY[i]].append(i)
    """
    def pos_in_grid(self, kp):

        posX = round((kp.pt[0] - self.mnMinX) * self.mfGridElementWidthInv)
        posY = round((kp.pt[1] - self.mnMinY) * self.mfGridElementHeightInv)

        if posX < 0 or posX >= self.FRAME_GRID_COLS or posY < 0 or posY >= self.FRAME_GRID_ROWS:
            return False, None, None

        return True, posX, posY

    def assign_features_to_grid(self):

        self.mGrid = [[[] for _ in range(self.N)] for _ in range(self.N)]
        for i in range(self.N):
            kp = self.mvKeys[i]
            bflag, nGridPosX, nGridPosY = self.pos_in_grid(kp)
            if bflag:
                self.mGrid[nGridPosX][nGridPosY].append(i)

    """

    def compute_stereo_matches(self):

        self.mvuRight = [-1] * self.N
        self.mvDepth = [-1] * self.N

        thOrbDist = (TH_HIGH  + TH_LOW)/2;
        nRows = self.mvImagePyramidLeft[0].shape[0]
        Nr = len(self.mvKeysRight)

        vRowIndices = [[] for _ in range(nRows)]
        for iR in range(Nr):
            kp = self.mvKeysRight[iR]
            kpY = kp.pt[1]
            r = 2.0 * self.mvScaleFactors[kp.octave]
            maxr = math.ceil(kpY + r)
            minr = math.floor(kpY - r)

            for yi in range(minr, maxr + 1):
                vRowIndices[yi].append(iR)

        minZ = self.mb
        minD = 0
        maxD = self.mbf / minZ;

        # disparity calculation
        vDistIdx = []
        for iL in range(self.N):

            kpL = self.mvKeys[iL]
            levelL = kpL.octave
            vL = kpL.pt[1]
            uL = kpL.pt[0]
            vCandidates = vRowIndices[int(vL)]

            if not vCandidates:
                continue

            minU = uL - maxD
            maxU = uL - minD

            if maxU < 0:
                continue

            bestDist = TH_HIGH
            bestIdxR = 0

            dL = self.mDescriptors[iL][:]
            for iC in vCandidates:

                kpR = self.mvKeysRight[iC]

                if (kpR.octave < levelL - 1) or (kpR.octave > levelL + 1):
                    continue

                uR = kpR.pt[0]
                if minU <= uR <= maxU:
                    dR = self.mDescriptorsRight[iC]
                    dist = self.descriptor_distance(dL, dR)
                    if dist < bestDist:
                        bestDist = dist
                        bestIdxR = iC

            #print("bestDist", bestDist)
            #print("thOrbDist", thOrbDist)
            if bestDist < thOrbDist:

                uR0 = self.mvKeysRight[bestIdxR].pt[0]
                scaleFactor = self.mvInvScaleFactors[kpL.octave]
                scaleduL = round(kpL.pt[0] * scaleFactor)
                scaledvL = round(kpL.pt[1] * scaleFactor)
                scaleduR0 = round(uR0 * scaleFactor)

                w = 5
                IL = self.mvImagePyramidLeft[kpL.octave][scaledvL - w:scaledvL + w + 1, scaleduL - w:scaleduL + w + 1]
                IL = IL.astype(np.float32)
                IL = IL - IL[w, w] * np.ones_like(IL, dtype=np.float32)

                bestDist = float("inf")
                bestincR = 0
                L = 5
                vDists = [0] * (2 * L + 1)

                iniu = scaleduR0 + L - w
                endu = scaleduR0 + L + w + 1
                if iniu < 0 or endu >= self.mvImagePyramidRight[kpL.octave].shape[1]: # check which dimention is cols
                    continue

                for incR in range(-L, L + 1):
                    IR = self.mvImagePyramidRight[kpL.octave][scaledvL - w:scaledvL + w + 1, scaleduR0 + incR - w:scaleduR0 + incR + w + 1]
                    IR = IR.astype(np.float32)
                    IR = IR - IR[w, w] * np.ones_like(IR, dtype=np.float32)

                    dist = np.sum(np.abs(IL - IR))
                    if dist < bestDist:
                        bestDist = dist
                        bestincR = incR

                    vDists[L + incR] = dist

                if bestincR == -L or bestincR == L:
                    continue

                # Sub-pixel match (Parabola fitting)
                dist1 = vDists[L + bestincR - 1]
                dist2 = vDists[L + bestincR]
                dist3 = vDists[L + bestincR + 1]

                deltaR = (dist1 - dist3) / (2.0 * (dist1 + dist3 - 2.0 * dist2))

                if deltaR < -1 or deltaR > 1:
                    continue

                # Re-scaled coordinate
                bestuR = self.mvScaleFactors[kpL.octave] * (scaleduR0 + bestincR + deltaR)

                disparity = uL - bestuR
                if minD <= disparity < maxD:
                    if disparity <= 0:
                        disparity = 0.01
                        bestuR = uL - 0.01

                    self.mvDepth[iL] = self.mbf / disparity
                    self.mvuRight[iL] = bestuR
                    #print(self.mvDepth[iL], self.mvuRight[iL])
                    vDistIdx.append((bestDist, iL))

    def unproject_stereo(self, i):
        z = self.mvDepth[i]
        if z > 0:
            u = self.mvKeys[i].pt[0]
            v = self.mvKeys[i].pt[1]
            x = (u - self.cx) * z * self.invfx
            y = (v - self.cy) * z * self.invfy
            x3Dc = np.array([[x], [y], [z]], dtype=np.float32)
            return self.mRwc @ x3Dc + self.mOw.reshape(3, 1)
        else:
            return None

    def undistort_keypoints(self):

        if self.mDistCoef[0][0] == 0:
           self.mvKeysUn=self.mvKeys
           return

        N = len(mvKeys)
        mat = np.zeros((N, 2), dtype=np.float32)
        for i, kp in enumerate(mvKeys):
            mat[i, 0] = kp.pt[0]
            mat[i, 1] = kp.pt[1]

        mat = mat.reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(mat, self.mK, self.mDistCoef, None, None, self.mK)
        undistorted = undistorted.reshape(-1, 2)

        mvKeysUn = []
        for i, kp in enumerate(mvKeys):
            new_kp = cv2.KeyPoint(
                x=undistorted[i, 0],
                y=undistorted[i, 1],
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
            mvKeysUn.append(new_kp)

        return mvKeysUn

    def descriptor_distance(self, a, b):
        xor = np.bitwise_xor(a, b)
        return sum(bin(byte).count('1') for byte in xor)

    def is_in_frustum(self, pMP, viewing_cos_limit):
        """
        Check if a MapPoint is in the frustum of the current frame.

        Args:
            pMP (MapPoint): The map point to check.
            viewing_cos_limit (float): The viewing cosine limit.

        Returns:
            bool: True if the map point is in the frustum, False otherwise.
        """
        pMP.mbTrackInView = False

        # 3D in absolute coordinates
        P = pMP.get_world_pos()

        # 3D in camera coordinates
        Pc = self.mRcw @ P + self.mtcw
        PcX = Pc[0]
        PcY = Pc[1]
        PcZ = Pc[2]

        # Check positive depth
        if PcZ < 0.0:
            return False

        # Project in image and check it is not outside
        invz = 1.0 / PcZ
        u = self.fx * PcX * invz + self.cx
        v = self.fy * PcY * invz + self.cy

        if u < self.mnMinX or u > self.mnMaxX:
            return False
        if v < self.mnMinY or v > self.mnMaxY:
           return False

        # Check distance is in the scale invariance region of the MapPoint
        max_distance = pMP.get_max_distance_invariance()
        min_distance = pMP.get_min_distance_invariance()
        PO = P - self.mOw
        dist = np.linalg.norm(PO)

        if dist < min_distance or dist > max_distance:
            return False

        # Check viewing angle
        Pn = pMP.get_normal()
        view_cos = (PO.T).dot(Pn) / dist

        if view_cos[0] < viewing_cos_limit:
            return False

        # Predict scale in the image
        n_predicted_level = pMP.predict_scale(dist, self)

        # Data used by the tracking
        pMP.mbTrackInView = True
        pMP.mTrackProjX = u
        pMP.mTrackProjXR = u - self.mbf * invz
        pMP.mTrackProjY = v
        pMP.mnTrackScaleLevel = n_predicted_level
        pMP.mTrackViewCos = view_cos
        return True


    def get_features_in_area(self, x, y, r, min_level, max_level):
        """
        Get the indices of features in a specified area and within scale levels.

        Args:
            x (float): X-coordinate of the center.
            y (float): Y-coordinate of the center.
            r (float): Radius around (x, y) to search.
            min_level (int): Minimum scale level to consider.
            max_level (int): Maximum scale level to consider.

        Returns:
            list: Indices of features within the area and scale levels.
        """
        v_indices = []

        n_min_cell_x = max(0, int((x - self.mnMinX - r) * self.mfGridElementWidthInv))
        if n_min_cell_x >= self.FRAME_GRID_COLS:
            return v_indices

        n_max_cell_x = min(self.FRAME_GRID_COLS - 1, int((x - self.mnMinX + r) * self.mfGridElementWidthInv))
        if n_max_cell_x < 0:
            return v_indices

        n_min_cell_y = max(0, int((y - self.mnMinY - r) * self.mfGridElementHeightInv))
        if n_min_cell_y >= self.FRAME_GRID_ROWS:
            return v_indices

        n_max_cell_y = min(self.FRAME_GRID_ROWS - 1, int((y - self.mnMinY + r) * self.mfGridElementHeightInv))
        if n_max_cell_y < 0:
            return v_indices

        b_check_levels = (min_level > 0) or (max_level >= 0)

        for ix in range(n_min_cell_x, n_max_cell_x + 1):
            for iy in range(n_min_cell_y, n_max_cell_y + 1):

                v_cell = self.mGrid[ix][iy]
                if not v_cell:
                    continue

                for g in v_cell:
                    kpUn = self.mvKeysUn[g]

                    if b_check_levels:
                        if kpUn.octave < min_level:
                            continue
                        if max_level >= 0 and kpUn.octave > max_level:
                            continue

                    distx = kpUn.pt[0] - x
                    disty = kpUn.pt[1] - y

                    if abs(distx) < r and abs(disty) < r:
                        v_indices.append(g)

        return v_indices


def compute_image_bounds(imLeft, mK, mDistCoef):

    if mDistCoef[0][0] != 0.0:
        mat = np.zeros((4, 2), dtype=np.float32)
        mat[0, 0] = 0.0
        mat[0, 1] = 0.0
        mat[1, 0] = imLeft.shape[1]
        mat[1, 1] = 0.0
        mat[2, 0] = 0.0
        mat[2, 1] = imLeft.shape[0]
        mat[3, 0] = imLeft.shape[1]
        mat[3, 1] = imLeft.shape[0]

        mat = mat.reshape(-1, 1, 2)
        mat = cv2.undistortPoints(mat, mK, mDistCoef, None, mK)
        mat = mat.reshape(-1, 2)

        mnMinX = min(mat[0, 0], mat[2, 0])
        mnMaxX = max(mat[1, 0], mat[3, 0])
        mnMinY = min(mat[0, 1], mat[1, 1])
        mnMaxY = max(mat[2, 1], mat[3, 1])
    else:
        mnMinX = 0.0
        mnMaxX = imLeft.shape[1]
        mnMinY = 0.0
        mnMaxY = imLeft.shape[0]

    return mnMinX, mnMaxX, mnMinY, mnMaxY

if __name__ == "__main__":
    import sys
    import yaml
    from pyDBoW.TemplatedVocabulary import TemplatedVocabulary
    from stereo_kitti import LoadImages
    import time

    sys.path.append("./pyORBExtractor/lib/")
    from pyORBExtractor import ORBextractor


    vocabulary = TemplatedVocabulary(k=5, L=3, weighting="TF_IDF", scoring="L1_NORM")
    vocabulary.load_from_text_file("./Vocabulary/ORBvoc.txt")

    with open("configs/KITTI00-02.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    leftImages, rightImages, timeStamps = LoadImages("00")
    nImages = len(leftImages)
    mleft = cv2.imread(leftImages[0], cv2.IMREAD_GRAYSCALE)
    mright = cv2.imread(rightImages[0], cv2.IMREAD_GRAYSCALE)
    timestamp = float(timeStamps[0])

    fx = cfg["Camera.fx"]
    fy = cfg["Camera.fy"]
    cx = cfg["Camera.cx"]
    cy = cfg["Camera.cy"]

    mk = np.eye(3, dtype=np.float32)
    mk[0][0] = fx
    mk[1][1] = fy
    mk[0][2] = cx
    mk[1][2] = cy


    mDistCoef = np.ones((1, 4), dtype=np.float32)
    mDistCoef[0][0] = cfg["Camera.k1"]
    mDistCoef[0][1] = cfg["Camera.k2"]
    mDistCoef[0][2] = cfg["Camera.p1"]
    mDistCoef[0][3] = cfg["Camera.p2"]

    mbf = cfg["Camera.bf"]
    mThDepth = mbf * cfg["ThDepth"] / fx;

    fps = cfg["Camera.fps"]
    if fps == 0:
        fps = 30

    mMinFrames = 0
    mMaxFrames = fps

    nRGB = cfg["Camera.RGB"]
    mbRGB = nRGB

    nFeatures = cfg["ORBextractor.nFeatures"]
    fScaleFactor = cfg["ORBextractor.scaleFactor"]
    nLevels = cfg["ORBextractor.nLevels"]
    fIniThFAST = cfg["ORBextractor.iniThFAST"]
    fMinThFAST = cfg["ORBextractor.minThFAST"]

    mORBExtractorLeft = ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
    mORBExtractorRight = ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)

    keypointsL = []
    keypointsL.append(cv2.KeyPoint(x=50.0, y=100.0, size=10.0, angle=45.0, response=0.8, octave=1, class_id=0))
    keypointsL.append(cv2.KeyPoint(x=150.0, y=200.0, size=12.5, angle=90.0, response=0.9, octave=2, class_id=1))
    keypointsL.append(cv2.KeyPoint(x=300.0, y=400.0, size=15.0, angle=-1.0, response=0.7, octave=3, class_id=2))


    keypointsR = []
    keypointsR.append(cv2.KeyPoint(x=54.0, y=140.0, size=13.0, angle=44.0, response=0.2, octave=1, class_id=0))
    keypointsR.append(cv2.KeyPoint(x=110.0, y=220.0, size=15.5, angle=20.0, response=0.4, octave=2, class_id=1))
    keypointsR.append(cv2.KeyPoint(x=200.0, y=440.0, size=17.0, angle=-11.0, response=0.6, octave=3, class_id=2))


    mnMinX, mnMaxX, mnMinY, mnMaxY = compute_image_bounds(mleft, mk, mDistCoef)

    FRAME_GRID_ROWS = 48
    FRAME_GRID_COLS = 64

    mfGridElementWidthInv = float(FRAME_GRID_COLS) / (mnMaxX - mnMinX)
    mfGridElementHeightInv = float(FRAME_GRID_ROWS) / (mnMaxY - mnMinY)

    invfx = 1.0 / fx
    invfy = 1.0 / fy

    # should be added to args
    frame_args = [fx, fy, cx, cy, invfx, invfy, mfGridElementWidthInv, mfGridElementHeightInv, mnMinX, mnMaxX, mnMinY, mnMaxY, FRAME_GRID_ROWS, FRAME_GRID_COLS]

    mTcw = np.eye(4, dtype=np.float32)

    # Rotation
    mTcw[0, 0] = 0.866
    mTcw[0, 1] = -0.5
    mTcw[1, 0] = 0.5
    mTcw[1, 1] = 0.866
    mTcw[2, 2] = 1.0

    # Set translation
    mTcw[0, 3] = 0.5
    mTcw[1, 3] = 0.3
    mTcw[2, 3] = 1.0

    print("start")
    for i in range(10):
        t = time.time()
        mFrame = Frame(mleft, mright, timestamp, mORBExtractorLeft, mORBExtractorRight, vocabulary, mk, mDistCoef, mbf, mThDepth, frame_args)
        print("time", time.time()-t)


