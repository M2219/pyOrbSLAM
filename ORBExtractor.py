import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import cv2
from pattern import pattern
import math
from collections import deque

HALF_PATCH_SIZE = 15
PATCH_SIZE = 31
EDGE_THRESHOLD = 19

def ic_angle(image, pt, umax):
    m_01 = 0.
    m_10 = 0.

    center_y, center_x = int(round(pt[1])), int(round(pt[0]))

    for u in range(-HALF_PATCH_SIZE, HALF_PATCH_SIZE + 1):
        m_10 += u * float(image[center_y, center_x + u])
    step = image.strides[0] // image.itemsize
    for v in range(1, HALF_PATCH_SIZE + 1):
        v_sum = 0
        d = umax[v]
        for u in range(-d, d + 1):
            val_plus = float(image[center_y + v, center_x + u])
            val_minus = float(image[center_y - v, center_x + u])
            v_sum += (val_plus - val_minus)
            m_10 += u * (val_plus + val_minus)
        m_01 += v * v_sum

    return (np.degrees(np.arctan2(m_01, m_10)) + 360.0) % 360.0

class ExtractorNode:
    def __init__(self, ul, ur, bl, br):
        self.UL = ul
        self.UR = ur
        self.BL = bl
        self.BR = br
        self.vKeys = []
        self.bNoMore = False

    def divide_node(self):
        """
        Divides the current node into four child nodes.
        Returns:
            n1, n2, n3, n4: Four child nodes.
        """
        half_x = math.ceil((self.UR[0] - self.UL[0]) / 2)
        half_y = math.ceil((self.BR[1] - self.UL[1]) / 2)

        n1 = ExtractorNode(self.UL, (self.UL[0] + half_x, self.UL[1]), (self.UL[0], self.UL[1] + half_y), (self.UL[0] + half_x, self.UL[1] + half_y))
        n2 = ExtractorNode(n1.UR, self.UR, n1.BR, (self.UR[0], self.UL[1] + half_y))
        n3 = ExtractorNode(n1.BL, n1.BR, self.BL, (n1.BR[0], self.BL[1]))
        n4 = ExtractorNode(n3.UR, n2.BR, n3.BR, self.BR)

        for kp in self.vKeys:
            if kp.pt[0] < n1.UR[0]:
                if kp.pt[1] < n1.BR[1]:
                    n1.vKeys.append(kp)
                else:
                    n3.vKeys.append(kp)
            else:
                if kp.pt[1] < n1.BR[1]:
                    n2.vKeys.append(kp)
                else:
                    n4.vKeys.append(kp)

        n1.bNoMore = len(n1.vKeys) == 1
        n2.bNoMore = len(n2.vKeys) == 1
        n3.bNoMore = len(n3.vKeys) == 1
        n4.bNoMore = len(n4.vKeys) == 1

        return n1, n2, n3, n4

class ORBExtractor:
    def __init__(self, nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST):
        self.nFeatures =  nFeatures
        self.nLevels = nLevels
        self.fIniThFAST = fIniThFAST
        self.fMinThFAST = fMinThFAST
        self.fScaleFactor = fScaleFactor
        self.pattern = pattern

        self.mvImagePyramid = []

        self.mvScaleFactor = [1.0]
        self.mvLevelSigma2 = [1.0]
        for i in range(1, self.nLevels):
            self.mvScaleFactor.append(self.mvScaleFactor[i - 1] * self.fScaleFactor)
            self.mvLevelSigma2.append(self.mvScaleFactor[i] * self.mvScaleFactor[i])

        self.mvInvScaleFactor = []
        self.mvInvLevelSigma2 = []
        for i in range(self.nLevels):
            self.mvInvScaleFactor.append(1.0 / self.mvScaleFactor[i])
            self.mvInvLevelSigma2.append(1.0 / self.mvLevelSigma2[i])

        self.factor = 1.0 / self.fScaleFactor;

        self.nDesiredFeaturesPerScale = self.nFeatures * (1 - self.factor) / (1 - pow(self.factor, self.nLevels))

        sumFeatures = 0
        self.mnFeaturesPerLevel = []
        for level in range(self.nLevels - 1):
            self.mnFeaturesPerLevel.append(round(self.nDesiredFeaturesPerScale))
            sumFeatures += self.mnFeaturesPerLevel[level]
            self.nDesiredFeaturesPerScale *= self.factor

        self.mnFeaturesPerLevel.append(max(self.nFeatures - sumFeatures, 0))

        npoints = 512
        pattern0 = self.pattern[:npoints]
        self.pattern.extend(pattern0)

        self.umax = [0] * (HALF_PATCH_SIZE + 1)

        vmax = math.floor(HALF_PATCH_SIZE * math.sqrt(2.0) / 2 + 1)
        vmin = math.ceil(HALF_PATCH_SIZE * math.sqrt(2.0) / 2)
        hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE

        for v in range(vmax + 1):
            self.umax[v] = round(math.sqrt(hp2 - v * v))

        v0 = 0
        for v in range(HALF_PATCH_SIZE, vmin - 1, -1):
            while self.umax[v0] == self.umax[v0 + 1]:
                v0 += 1
            self.umax[v] = v0
            v0 += 1

    def operator(self, image):

        assert len(image.shape) == 2

        image_pyramid = self.compute_pyramid(image)

        all_keypoints = self.compute_keypoints_octree(image_pyramid)

        nkeypoints = sum(len(keypoints) for keypoints in all_keypoints)
        if nkeypoints == 0:
            print("nkeypoints is None")
            return
        else:
            desc_t = []

        self.descriptors = []
        self.keypoints = []
        for level, kps in enumerate(all_keypoints):

            nkeypointsLevel = len(kps)
            if nkeypointsLevel == 0:
                continue

            #workingMat = image_pyramid[level].copy() #  workingMat = self.mvImagePyramid[level].copy() for optimization
            working_mat = cv2.GaussianBlur(image_pyramid[level], (7, 7), 2, 2, cv2.BORDER_REFLECT_101)
            desc_computed = self.compute_descriptors(working_mat, kps, self.pattern)

            if level != 0:
                scale = self.mvScaleFactor[level]
                for kp in kps:
                    kp.pt = (kp.pt[0] * scale, kp.pt[1] * scale)

            self.keypoints.extend(kps)
            desc_t.append(desc_computed)

        self.descriptors = np.concatenate(desc_t)
        return self.keypoints, self.descriptors

    def compute_pyramid(self, image):
        for level in range(self.nLevels):

            scale = self.mvInvScaleFactor[level]

            scaled_width = int(round(image.shape[1] * scale))
            scaled_height = int(round(image.shape[0] * scale))
            if level != 0:
                temp = cv2.resize(self.mvImagePyramid[level -1], (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
                self.mvImagePyramid.append(temp)

            else:
                self.mvImagePyramid = [image]

        return self.mvImagePyramid

    def compute_keypoints_octree(self, image_pyramid):
        W = 30
        all_keypoints = []
        for level in range(self.nLevels):
            min_border_x = EDGE_THRESHOLD - 3
            min_border_y = min_border_x
            max_border_x = image_pyramid[level].shape[1] - EDGE_THRESHOLD + 3
            max_border_y = image_pyramid[level].shape[0] - EDGE_THRESHOLD + 3

            to_distribute_keys = []

            width = max_border_x - min_border_x
            height = max_border_y - min_border_y

            n_cols = int(width // W)
            n_rows = int(height // W)
            w_cell = math.ceil(width / n_cols)
            h_cell = math.ceil(height / n_rows)

            for i in range(n_rows):
                ini_y = min_border_y + i * h_cell
                max_y = ini_y + h_cell + 6

                if ini_y >= max_border_y - 3:
                    continue
                if max_y > max_border_y:
                    max_y = max_border_y

                for j in range(n_cols):
                    ini_x = min_border_x + j * w_cell
                    max_x = ini_x + w_cell + 6

                    if ini_x >= max_border_x - 6:
                        continue
                    if max_x > max_border_x:
                        max_x = max_border_x

                    cell_region = image_pyramid[level][int(ini_y):int(max_y), int(ini_x):int(max_x)]
                    keypoints_cell = []

                    fast = cv2.FastFeatureDetector_create(self.fIniThFAST, True)
                    keypoints_cell = fast.detect(cell_region)

                    if not keypoints_cell:
                        fast = cv2.FastFeatureDetector_create(self.fMinThFAST, True)
                        keypoints_cell = fast.detect(cell_region)

                    if keypoints_cell:
                        for kp in keypoints_cell:
                            kp.pt = (kp.pt[0] + j * w_cell, kp.pt[1] + i * h_cell)
                            to_distribute_keys.append(kp)

            keypoints = self.distribute_octree(to_distribute_keys, min_border_x, max_border_x, min_border_y, max_border_y, self.mnFeaturesPerLevel[level], level)
            ######################################################
            #     keypoint python has two more features          #
            ######################################################
            all_keypoints.append(keypoints)

            scaled_patch_size = PATCH_SIZE * self.mvScaleFactor[level]
            for kp in keypoints:
                kp.pt = (kp.pt[0] + min_border_x, kp.pt[1] + min_border_y)
                kp.octave = level
                kp.size = scaled_patch_size

        for level in range(self.nLevels):
            all_keypoints[level] = self.compute_orientation(image_pyramid[level], all_keypoints[level], self.umax)

        return all_keypoints

    def distribute_octree(self, v_to_distribute_keys, min_x, max_x, min_y, max_y, n, level):
        n_ini = round((max_x - min_x) / (max_y - min_y))
        hx = (max_x - min_x) / n_ini
        l_nodes = deque()
        vp_inv_nodes = deque()

        for i in range(n_ini):
            ul = (int(i * hx), 0)
            ur = (int((i + 1) * hx), 0)
            bl = (int(i * hx), max_y - min_y)
            br = (int((i + 1) * hx), max_y - min_y)

            node = ExtractorNode(ul, ur, bl, br)
            l_nodes.append(node)
            vp_inv_nodes.appendleft(node)

        for kp in v_to_distribute_keys:
            idx = int((kp.pt[0] - min_x) / hx)
            if 0 <= idx < len(vp_inv_nodes):
                vp_inv_nodes[idx].vKeys.append(kp)

        for node in list(l_nodes):
            if len(node.vKeys) == 1:
                node.bNoMore = True
            elif len(node.vKeys) == 0:
                l_nodes.remove(node)


        b_finish = False
        iteration = 0

        while not b_finish:

            iteration += 1
            n_to_expand = 0
            v_size_and_pointer_to_node = []
            prev_size = len(l_nodes)

            for lit in list(l_nodes):
                if lit.bNoMore:
                    continue

                else:
                    n1, n2, n3, n4 = lit.divide_node()
                    for child in [n1, n2, n3, n4]:
                        if len(child.vKeys) > 0:
                            l_nodes.appendleft(child)
                            if len(child.vKeys) > 1:
                                n_to_expand += 1
                                v_size_and_pointer_to_node.append((len(child.vKeys), child))

                l_nodes.remove(lit)

            if len(l_nodes) >= n or len(l_nodes) == prev_size:
                b_finish = True

            elif len(l_nodes) + n_to_expand * 3 > n:

                while not b_finish:
                    prev_size = len(l_nodes)

                    v_prev_size_and_pointer_to_node = sorted(v_size_and_pointer_to_node, reverse=True, key=lambda x: x[0])
                    v_size_and_pointer_to_node.clear()

                    for size, node in v_prev_size_and_pointer_to_node:
                        n1, n2, n3, n4 = node.divide_node()
                        for child in [n1, n2, n3, n4]:
                            if len(child.vKeys) > 0:
                                l_nodes.appendleft(child)
                                if len(child.vKeys) > 1:
                                    v_size_and_pointer_to_node.append((len(child.vKeys), child))


                        l_nodes.remove(node)
                        if len(l_nodes) >= n:
                            break

                    if len(l_nodes) >= n or len(l_nodes) == prev_size:
                        b_finish = True

        result_keys = []
        for node in l_nodes:
            if len(node.vKeys) > 0:
                best_kp = max(node.vKeys, key=lambda kp: kp.response)
                result_keys.append(best_kp)

        return result_keys

    def compute_orientation(self, image, keypoints, umax):
        keys = []
        for keypoint in keypoints:
            keypoint.angle = ic_angle(image, keypoint.pt, umax)
            keys.append(keypoint)
        return keys

    def compute_descriptors(self, image, keypoints, pattern):

        desc = np.zeros((len(keypoints), 32), dtype=np.uint8)
        for i, kp in enumerate(keypoints):
            desc[i] = self.compute_orb_descriptor(kp, image, pattern)

        return desc

    def compute_orb_descriptor(self, kpt, img, pattern):

        factorPI = np.pi / 180.0
        angle = kpt.angle * factorPI
        a = np.cos(angle)
        b = np.sin(angle)
        center_y, center_x = int(round(kpt.pt[1])), int(round(kpt.pt[0]))

        desc = np.zeros(32, dtype=np.uint8)

        def get_value(idx):
            dy = round(pattern[idx][0] * b + pattern[idx][1] * a)
            dx = round(pattern[idx][0] * a - pattern[idx][1] * b)
            y, x = center_y + dy, center_x + dx
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                return img[y, x]
            else:
                return 0

        j = 0
        for i in range(32):

            t0 = get_value(0 + j); t1 = get_value(1 + j);
            val = t0 < t1;
            t0 = get_value(2 + j); t1 = get_value(3 + j);
            val |= (t0 < t1) << 1;
            t0 = get_value(4 + j); t1 = get_value(5 + j);
            val |= (t0 < t1) << 2;
            t0 = get_value(6 + j); t1 = get_value(7 + j);
            val |= (t0 < t1) << 3;
            t0 = get_value(8 + j); t1 = get_value(9 + j);
            val |= (t0 < t1) << 4;
            t0 = get_value(10 + j); t1 = get_value(11 + j);
            val |= (t0 < t1) << 5;
            t0 = get_value(12 + j); t1 = get_value(13 + j);
            val |= (t0 < t1) << 6;
            t0 = get_value(14 + j); t1 = get_value(15 + j);
            val |= (t0 < t1) << 7;

            desc[i] = val
            j = j + 8

        return desc

    def compute_features_per_level(self):
        factor = 1.0 / self.scale_factors[1]
        n_desired_features = self.nFeatures * (1 - factor) / (1 - factor ** self.nLevels)
        features_per_level = []
        for i in range(self.nLevels - 1):
            features = round(n_desired_features)
            features_per_level.append(features)
            n_desired_features *= factor
        features_per_level.append(max(self.nFeatures - sum(features_per_level), 0))
        return features_per_level


if __name__ == "__main__":

    import yaml

    with open("configs/KITTI00-02.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    nFeatures = cfg["ORBextractor.nFeatures"]
    fScaleFactor = cfg["ORBextractor.scaleFactor"]
    nLevels = cfg["ORBextractor.nLevels"]
    fIniThFAST = cfg["ORBextractor.iniThFAST"]
    fMinThFAST = cfg["ORBextractor.minThFAST"]

    ORBExtractor = ORBExtractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST)

    im_gray = cv2.imread('./images/image0.png', cv2.IMREAD_GRAYSCALE)
    mask = im_gray // 11
    keypoints = []

    # Add dummy keypoints to the list
    keypoints.append(cv2.KeyPoint(x=50.0, y=100.0, size=10.0, angle=45.0, response=0.8, octave=1, class_id=0))
    keypoints.append(cv2.KeyPoint(x=150.0, y=200.0, size=12.5, angle=90.0, response=0.9, octave=2, class_id=1))
    keypoints.append(cv2.KeyPoint(x=300.0, y=400.0, size=15.0, angle=-1.0, response=0.7, octave=3, class_id=2))

    keyes, descriptors = ORBExtractor.operator(im_gray)


    """

    fx = cfg["Camera.fx"];
    fy = cfg["Camera.fy"];
    cx = cfg["Camera.cx"];
    cy = cfg["Camera.cy"];

    mk = np.eye(3, dtype=np.float32)
    mk[0][0] = fx
    mk[1][1] = fy
    mk[0][2] = cx
    mk[1][2] = cy

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    cv::Mat mDistCoef;
    DistCoef.copyTo(mDistCoef);

    float mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    int mMinFrames = 0;
    int mMaxFrames = fps;
    """







