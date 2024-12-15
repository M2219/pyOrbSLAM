import threading
import math

import numpy as np
import cv2
from ORBMatcher import ORBMatcher
from Convertor import Convertor

class Frame:

    nNextId = 0
    def __init__(self, mleft, mright, timestamp, mpORBextractorLeft, mpORBextractorRight,
                   mpVocabulary, mK, mDistCoef, mbf, mThDepth, mTcw, frame_args):

        self.mTcw = mTcw

        if self.mTcw is not None:
            self.set_pose(self.mTcw)

        # self.args = args
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

        self.mnScaleLevels = mpORBextractorLeft.nLevels
        self.mfScaleFactor = mpORBextractorLeft.fScaleFactor
        self.mfLogScaleFactor = np.log(self.mfScaleFactor)
        self.mvScaleFactors = mpORBextractorLeft.mvScaleFactor
        self.mvInvScaleFactors = mpORBextractorLeft.mvInvScaleFactor
        self.mvLevelSigma2 = mpORBextractorLeft.mvLevelSigma2
        self.mvInvLevelSigma2 = mpORBextractorLeft.mvInvLevelSigma2

        self.mb = self.mbf / self.mK[0][0]

        #if mTcw is not None:
	        #    self.set_pose(mTcw)

        threadLeft = threading.Thread(target=self.ExtractORB, args=(0, mleft))
        threadRight = threading.Thread(target=self.ExtractORB, args=(1, mright))
        threadLeft.start()
        threadRight.start()
        threadLeft.join()
        threadRight.join()

        self.N = len(self.mvKeys)

        self.undistort_keypoints()

        self.ORBM = ORBMatcher()
        self.compute_stereo_matches()

        self.mvpMapPoints = {}
        self.mvbOutlier = []

        self.assign_features_to_grid()

        self.mnId = Frame.nNextId
        Frame.nNextId += 1

    def ExtractORB(self, flag, image):
        if flag == 0:
            self.mvKeys, self.mDescriptors = self.mpORBextractorLeft.operator(image)

        elif flag == 1:
            self.mvKeysRight, self.mDescriptorsRight = self.mpORBextractorRight.operator(image)

    def Compute_BoW(self):

        if self.mBoWVec is None:
            vCurrentDesc = Convertor.to_descriptor_vector(self.mDescriptors)
            self.mBowVec, self.mFeatVec = self.mpORBvocabulary.transform(vCurrentDesc, 4)


    def set_pose(self, Tcw_):
        self.mTcw = Tcw_.copy()
        self.update_pose_matrices()

    def update_pose_matrices(self):
        self.mRcw = self.mTcw[:3, :3]
        self.mRwc = self.mRcw.T
        self.mtcw = self.mTcw[:3, 3].reshape(3, 1)
        self.mOw = -np.dot(self.mRwc, self.mtcw)

    def PosInGrid(self, kp):

        posX = round((kp.pt[0] - self.mnMinX) * self.mfGridElementWidthInv)
        posY = round((kp.pt[1] - self.mnMinY) * self.mfGridElementHeightInv)

        # Ensure the keypoint is within the grid bounds
        if posX < 0 or posX >= self.FRAME_GRID_COLS or posY < 0 or posY >= self.FRAME_GRID_ROWS:
            return False, None, None

        return True, posX, posY

    def assign_features_to_grid(self):

        mGrid = []
        mg = np.zeros((self.FRAME_GRID_COLS, self.FRAME_GRID_ROWS), dtype=np.float32)
        for i in range(self.N):
            kp = self.mvKeys[i]
            bflag, nGridPosX, nGridPosY = self.PosInGrid(kp)

            if bflag:
                mGrid.append(mg[nGridPosX][nGridPosY])

    def compute_stereo_matches(self):

        self.mvuRight = [0] * self.N
        self.mvDepth = [-1] * self.N

        thOrbDist = (self.ORBM.TH_HIGH  + self.ORBM.TH_LOW)/2;
        nRows = self.mpORBextractorLeft.mvImagePyramid[0].shape[0]
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

            bestDist = self.ORBM.TH_HIGH
            bestIdxR = 0

            dL = self.mDescriptors[iL][:]
            for iC in vCandidates:

                kpR = self.mvKeysRight[iC]

                if (kpR.octave < levelL - 1) or (kpR.octave > levelL + 1):
                    continue

                uR = kpR.pt[0]
                if minU <= uR <= maxU:
                    dR = self.mDescriptorsRight[iC]
                    dist = self.ORBM.descriptor_distance(dL, dR)
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
                IL = self.mpORBextractorLeft.mvImagePyramid[kpL.octave][scaledvL - w:scaledvL + w + 1, scaleduL - w:scaleduL + w + 1]
                IL = IL.astype(np.float32)
                IL = IL - IL[w, w] * np.ones_like(IL, dtype=np.float32)

                bestDist = float("inf")
                bestincR = 0
                L = 5
                vDists = [0] * (2 * L + 1)

                iniu = scaleduR0 + L - w
                endu = scaleduR0 + L + w + 1
                if iniu < 0 or endu >= self.mpORBextractorRight.mvImagePyramid[kpL.octave].shape[1]: # check which dimention is cols
                    continue

                for incR in range(-L, L + 1):
                    IR = self.mpORBextractorRight.mvImagePyramid[kpL.octave][scaledvL - w:scaledvL + w + 1, scaleduR0 + incR - w:scaleduR0 + incR + w + 1]
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

    import yaml
    from pyDBoW.TemplatedVocabulary import TemplatedVocabulary
    from ORBExtractor import ORBExtractor
    from stereo_kitti import LoadImages

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

    mORBExtractorLeft = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)
    mORBExtractorRight = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)

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


    mFrame = Frame(mleft, mright, timestamp, mORBExtractorLeft, mORBExtractorRight, vocabulary, mk, mDistCoef, mbf, mThDepth, mTcw, frame_args)





