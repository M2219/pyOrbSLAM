import numpy as np
import math

class PnPsolver:
    def __init__(self, F, vpMapPointMatches):
        self.maximum_number_of_correspondences = 0
        self.number_of_correspondences = 0
        self.mnInliersi = 0
        self.mnIterations = 0
        self.mnBestInliers = 0
        self.N = 0

        self.mvpMapPointMatches = vpMapPointMatches
        self.mvP2D = []
        self.mvSigma2 = []
        self.mvP3Dw = []
        self.mvKeyPointIndices = []
        self.mvAllIndices = []

        idx = 0
        for i, pMP in enumerate(vpMapPointMatches):
            if pMP and not pMP.is_bad():
                kp = F.mvKeysUn[i]

                self.mvP2D.append(kp.pt)
                self.mvSigma2.append(F.mvLevelSigma2[kp.octave])

                Pos = pMP.get_world_pos()
                self.mvP3Dw.append(np.array([Pos[0], Pos[1], Pos[2]], dtype=np.float32))

                self.mvKeyPointIndices.append(i)
                self.mvAllIndices.append(idx)

                idx += 1

        self.fu = F.fx
        self.fv = F.fy
        self.uc = F.cx
        self.vc = F.cy

        self.set_ransac_parameters(probability=0.99, minInliers=8 , maxIterations=300, minSet=4, epsilon=0.4, th2=5.991)

    def set_ransac_parameters(self, probability, minInliers, maxIterations, minSet, epsilon, th2):
        self.mRansacProb = probability
        self.mRansacMinInliers = minInliers
        self.mRansacMaxIts = maxIterations
        self.mRansacEpsilon = epsilon
        self.mRansacMinSet = minSet

        self.N = len(self.mvP2D)

        self.mvbInliersi = [False] * self.N

        nMinInliers = int(self.N * self.mRansacEpsilon)
        if nMinInliers < self.mRansacMinInliers:
            nMinInliers = self.mRansacMinInliers
        if nMinInliers < self.mRansacMinSet:
            nMinInliers = self.mRansacMinSet
        self.mRansacMinInliers = nMinInliers

        if self.mRansacEpsilon < float(self.mRansacMinInliers) / self.N:
            self.mRansacEpsilon = float(self.mRansacMinInliers) / self.N

        if self.mRansacMinInliers == self.N:
            nIterations = 1

        else:
            nIterations = math.ceil(math.log(1 - self.mRansacProb) / math.log(1 - math.pow(self.mRansacEpsilon, 3)))

        self.mRansacMaxIts = nIterations

        self.mvMaxError = [sigma2 * th2 for sigma2 in self.mvSigma2]

    def find(self):
        self.mBestTcw, bNoMore, vbInliers12, nInliers = self.iterate(self.mRansacMaxIts)
        return  vbInliers12, nInliers

    def iterate(self, nIterations):
        bNoMore = False
        vbInliers = []
        nInliers = 0

        self.set_maximum_number_of_correspondences(self.mRansacMinSet)
        if self.N < self.mRansacMinInliers:
            bNoMore = True
            return None, bNoMore, vbInliers, nInliers

        vAvailableIndices = []

        nCurrentIterations = 0
        while self.mnIterations < self.mRansacMaxIts or nCurrentIterations < nIterations:
            nCurrentIterations += 1
            self.mnIterations += 1
            self.reset_correspondences()

            vAvailableIndices = self.mvAllIndices.copy()

            for _ in range(self.mRansacMinSet):
                randi = np.random.randint(0, len(vAvailableIndices))

                idx = vAvailableIndices[randi]

                self.add_correspondence(
                    self.mvP3Dw[idx][0], self.mvP3Dw[idx][1], self.mvP3Dw[idx][2],
                    self.mvP2D[idx][0], self.mvP2D[idx][1]
                )

                vAvailableIndices[randi] = vAvailableIndices[-1]
                vAvailableIndices.pop()

            self.compute_pose()

            self.check_inliers()
            if self.mnInliersi >= self.mRansacMinInliers:
                if self.mnInliersi > self.mnBestInliers:
                    self.mvbBestInliers = self.mvbInliersi.copy()
                    self.mnBestInliers = self.mnInliersi

                    Rcw = np.array(self.mRi).reshape(3, 3).astype(np.float32)
                    tcw = np.array(self.mti).reshape(3, 1).astype(np.float32)
                    self.mBestTcw = np.eye(4, dtype=np.float32)
                    self.mBestTcw[:3, :3] = Rcw
                    self.mBestTcw[:3, 3] = tcw.flatten()

                if self.refine():
                    nInliers = self.mnRefinedInliers
                    vbInliers.extend([False] * len(self.mvpMapPointMatches))
                    for i in range(self.N):
                        if self.mvbRefinedInliers[i]:
                            vbInliers[self.mvKeyPointIndices[i]] = True
                    return self.mRefinedTcw.copy(), bNoMore, vbInliers, nInliers

        if self.mnIterations >= self.mRansacMaxIts:
            bNoMore = True
            if self.mnBestInliers >= self.mRansacMinInliers:
                nInliers = self.mnBestInliers
                vbInliers.extend([False] * len(self.mvpMapPointMatches))
                for i in range(self.N):
                    if self.mvbBestInliers[i]:
                        vbInliers[self.mvKeyPointIndices[i]] = True
                return self.mBestTcw.copy(), bNoMore, vbInliers, nInliers

        return None, bNoMore, vbInliers, nInliers

    def refine(self):
        v_indices = [i for i, is_inlier in enumerate(self.mvbBestInliers) if is_inlier]

        self.set_maximum_number_of_correspondences(len(v_indices))
        self.reset_correspondences()

        for idx in v_indices:
            self.add_correspondence(
                self.mvP3Dw[idx][0], self.mvP3Dw[idx][1], self.mvP3Dw[idx][2],
                self.mvP2D[idx][0], self.mvP2D[idx][1]
            )

        self.compute_pose()

        self.check_inliers()

        self.mnRefinedInliers = self.mnInliersi
        self.mvbRefinedInliers = self.mvbInliersi.copy()

        if self.mnInliersi > self.mRansacMinInliers:
            Rcw = np.array(self.mRi).reshape(3, 3).astype(np.float32)
            tcw = np.array(self.mti).reshape(3, 1).astype(np.float32)

            self.mRefinedTcw = np.eye(4, dtype=np.float32)
            self.mRefinedTcw[:3, :3] = Rcw
            self.mRefinedTcw[:3, 3] = tcw.flatten()

            return True

        return False

    def check_inliers(self):
        self.mnInliersi = 0

        for i in range(self.N):
            P3Dw = self.mvP3Dw[i]
            P2D = self.mvP2D[i]
            Xc = (
                self.mRi[0][0] * P3Dw[0]
                + self.mRi[0][1] * P3Dw[1]
                + self.mRi[0][2] * P3Dw[2]
                + self.mti[0]
            )
            Yc = (
                self.mRi[1][0] * P3Dw[0]
                + self.mRi[1][1] * P3Dw[1]
                + self.mRi[1][2] * P3Dw[2]
                + self.mti[1]
            )
            invZc = 1 / (
                self.mRi[2][0] * P3Dw[0]
                + self.mRi[2][1] * P3Dw[1]
                + self.mRi[2][2] * P3Dw[2]
                + self.mti[2]
            )

            ue = self.uc + self.fu * Xc * invZc
            ve = self.vc + self.fv * Yc * invZc

            distX = P2D[0] - ue
            distY = P2D[1] - ve
            error2 = distX * distX + distY * distY

            if error2 < self.mvMaxError[i]:
                self.mvbInliersi[i] = True
                self.mnInliersi += 1
            else:
                self.mvbInliersi[i] = False

    def set_maximum_number_of_correspondences(self, n):
        if self.maximum_number_of_correspondences < n:
            self.pws = np.zeros((3 * n,), dtype=np.float64)
            self.us = np.zeros((2 * n,), dtype=np.float64)
            self.alphas = np.zeros((4 * n,), dtype=np.float64)
            self.pcs = np.zeros((3 * n,), dtype=np.float64)
            self.maximum_number_of_correspondences = n

    def reset_correspondences(self):
        self.number_of_correspondences = 0

    def add_correspondence(self, X, Y, Z, u, v):
        index_3d = 3 * self.number_of_correspondences
        index_2d = 2 * self.number_of_correspondences

        self.pws[index_3d] = X
        self.pws[index_3d + 1] = Y
        self.pws[index_3d + 2] = Z

        self.us[index_2d] = u
        self.us[index_2d + 1] = v

        self.number_of_correspondences += 1

    def choose_control_points(self):

        self.cws = np.zeros((self.number_of_correspondences, 3), dtype=np.float64)
        for i in range(self.number_of_correspondences):
            for j in range(3):
                self.cws[0, j] += self.pws[3 * i + j]

        self.cws[0] /= self.number_of_correspondences
        PW0 = np.zeros((self.number_of_correspondences, 3), dtype=np.float64)
        for i in range(self.number_of_correspondences):
            for j in range(3):
                PW0[i, j] = self.pws[3 * i + j] - self.cws[0, j]

        PW0tPW0 = PW0.T @ PW0

        uc, dc, vt = np.linalg.svd(PW0tPW0)
        uct = uc.T
        for i in range(1, 4):
            k = np.sqrt(dc[i - 1] / self.number_of_correspondences)
            for j in range(3):
                self.cws[i, j] = self.cws[0, j] + k * uct[i - 1, j]

    def compute_barycentric_coordinates(self):
        cc = np.zeros((3, 3))
        for i in range(3):
            for j in range(1, 4):
                cc[i, j - 1] = self.cws[j][i] - self.cws[0][i]

        cc_inv = np.linalg.inv(cc)

        for i in range(self.number_of_correspondences):
            pi = self.pws[3 * i:3 * i + 3]
            a = self.alphas[4 * i:4 * i + 4]

            for j in range(3):
                a[1 + j] = (
                    cc_inv[j, 0] * (pi[0] - self.cws[0][0]) +
                    cc_inv[j, 1] * (pi[1] - self.cws[0][1]) +
                    cc_inv[j, 2] * (pi[2] - self.cws[0][2])
                )
            a[0] = 1.0 - a[1] - a[2] - a[3]

    def fill_M(self, M, row, alphas, u, v):
        M1 = M[row]
        M2 = M[row + 1]

        for i in range(4):
            M1[3 * i] = alphas[i] * self.fu
            M1[3 * i + 1] = 0.0
            M1[3 * i + 2] = alphas[i] * (self.uc - u)

            M2[3 * i] = 0.0
            M2[3 * i + 1] = alphas[i] * self.fv
            M2[3 * i + 2] = alphas[i] * (self.vc - v)

    def compute_ccs(self, betas, ut):

        self.ccs = np.zeros((4, 3), dtype=np.float64)

        for i in range(4):
            v = ut[11 - i, :]
            for j in range(4):
                for k in range(3):
                    self.ccs[j][k] += betas[i] * v[3 * j + k]

    def compute_pcs(self):
        for i in range(self.number_of_correspondences):
            a = self.alphas[4 * i:4 * i + 4]
            pc = self.pcs[3 * i:3 * i + 3]

            for j in range(3):
                pc[j] = a[0] * self.ccs[0][j] + a[1] * self.ccs[1][j] + a[2] * self.ccs[2][j] + a[3] * self.ccs[3][j]

    def solve_for_sign(self):
        if self.pcs[2] < 0.0:
            self.ccs[:, :] = -self.ccs
            self.pcs = -self.pcs

    def reprojection_error(self, R, t):
        sum2 = 0.0
        for i in range(self.number_of_correspondences):
            pw = self.pws[3 * i: 3 * i + 3]
            Xc = self.dot(R[0], pw) + t[0]
            Yc = self.dot(R[1], pw) + t[1]
            inv_Zc = 1.0 / (self.dot(R[2], pw) + t[2])
            ue = self.uc + self.fu * Xc * inv_Zc
            ve = self.vc + self.fv * Yc * inv_Zc
            u, v = self.us[2 * i], self.us[2 * i + 1]
            sum2 += np.sqrt((u - ue) ** 2 + (v - ve) ** 2)

        return sum2 / self.number_of_correspondences

    def estimate_R_and_t(self, R, t):
        pc0 = np.zeros(3, dtype=np.float64)
        pw0 = np.zeros(3, dtype=np.float64)

        for i in range(self.number_of_correspondences):
            pc = self.pcs[3 * i:3 * i + 3]
            pw = self.pws[3 * i:3 * i + 3]
            pc0 += pc
            pw0 += pw

        pc0 /= self.number_of_correspondences
        pw0 /= self.number_of_correspondences

        ABt = np.zeros((3, 3), dtype=np.float64)
        for i in range(self.number_of_correspondences):
            pc = self.pcs[3 * i:3 * i + 3]
            pw = self.pws[3 * i:3 * i + 3]

            for j in range(3):
                ABt[j, 0] += (pc[j] - pc0[j]) * (pw[0] - pw0[0])
                ABt[j, 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1])
                ABt[j, 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2])

        U, D, Vt = np.linalg.svd(ABt, full_matrices=True)
        R[:3, :3] = np.dot(U, Vt)

        if np.linalg.det(R[:3, :3]) < 0:
            R[:, 2] *= -1

        t[:] = pc0 - np.dot(R[:3, :3], pw0)

    def compute_R_and_t(self, ut, betas, R, t):
        self.compute_ccs(betas, ut)
        self.compute_pcs()

        self.solve_for_sign()

        self.estimate_R_and_t(R, t)
        return self.reprojection_error(R, t)

    def compute_pose(self):
        self.choose_control_points()
        self.compute_barycentric_coordinates()

        M = np.zeros((2 * self.number_of_correspondences, 12), dtype=np.float64)
        for i in range(self.number_of_correspondences):
            self.fill_M(M, 2 * i, self.alphas[4 * i: 4 * i + 4], self.us[2 * i], self.us[2 * i + 1])

        MtM = np.dot(M.T, M)
        U, f, Vt = np.linalg.svd(MtM, full_matrices=True)

        L_6x10 = np.zeros((6, 10), dtype=np.float64)
        rho = np.zeros((6,), dtype=np.float64)

        self.compute_l_6x10(U.T, L_6x10)
        self.compute_rho(rho)

        Betas = np.zeros((4, 4), dtype=np.float64)
        rep_errors = np.zeros((4,), dtype=np.float64)
        Rs = np.zeros((4, 3, 3), dtype=np.float64)
        ts = np.zeros((4, 3), dtype=np.float64)

        self.find_betas_approx_1(L_6x10, rho, Betas[1])
        self.gauss_newton(L_6x10, rho, Betas[1])
        rep_errors[1] = self.compute_R_and_t(U.T, Betas[1], Rs[1], ts[1])

        self.find_betas_approx_2(L_6x10, rho, Betas[2])
        self.gauss_newton(L_6x10, rho, Betas[2])
        rep_errors[2] = self.compute_R_and_t(U.T, Betas[2], Rs[2], ts[2])

        self.find_betas_approx_3(L_6x10, rho, Betas[3])
        self.gauss_newton(L_6x10, rho, Betas[3])
        rep_errors[3] = self.compute_R_and_t(U.T, Betas[3], Rs[3], ts[3])

        N = 1
        if rep_errors[2] < rep_errors[1]: N = 2
        if rep_errors[3] < rep_errors[N]: N = 3

        self.mRi = Rs[N]
        self.mti = ts[N]

        return rep_errors[N]

    def dist2(self, p1, p2):
        return np.sum((np.array(p1) - np.array(p2)) ** 2)

    def dot(self, v1, v2):
        return np.dot(np.array(v1), np.array(v2))

    def print_pose(self, R, t):
        for i in range(3):
            print(f"{R[i, 0]} {R[i, 1]} {R[i, 2]} {t[i]}")

    def find_betas_approx_1(self, L_6x10, Rho, betas):
        L_6x4 = L_6x10[:, [0, 1, 3, 6]]
        b4 = np.linalg.lstsq(L_6x4, Rho, rcond=None)[0]

        if b4[0] < 0:
            betas[0] = np.sqrt(-b4[0])
            betas[1] = -b4[1] / betas[0]
            betas[2] = -b4[2] / betas[0]
            betas[3] = -b4[3] / betas[0]
        else:
            betas[0] = np.sqrt(b4[0])
            betas[1] = b4[1] / betas[0]
            betas[2] = b4[2] / betas[0]
            betas[3] = b4[3] / betas[0]

    def find_betas_approx_2(self, L_6x10, Rho, betas):
        L_6x3 = L_6x10[:, [0, 1, 2]]
        b3 = np.linalg.lstsq(L_6x3, Rho, rcond=None)[0]

        if b3[0] < 0:
            betas[0] = np.sqrt(-b3[0])
            betas[1] = np.sqrt(-b3[2]) if b3[2] < 0 else 0.0
        else:
            betas[0] = np.sqrt(b3[0])
            betas[1] = np.sqrt(b3[2]) if b3[2] > 0 else 0.0

        if b3[1] < 0:
            betas[0] = -betas[0]

        betas[2] = 0.0
        betas[3] = 0.0

    def find_betas_approx_3(self, L_6x10, Rho, betas):
        L_6x5 = L_6x10[:, [0, 1, 2, 3, 4]]
        b5 = np.linalg.lstsq(L_6x5, Rho, rcond=None)[0]

        if b5[0] < 0:
            betas[0] = np.sqrt(-b5[0])
            betas[1] = np.sqrt(-b5[2]) if b5[2] < 0 else 0.0
        else:
            betas[0] = np.sqrt(b5[0])
            betas[1] = np.sqrt(b5[2]) if b5[2] > 0 else 0.0

        if b5[1] < 0:
            betas[0] = -betas[0]

        betas[2] = b5[3] / betas[0]
        betas[3] = 0.0

    def compute_l_6x10(self, ut, l_6x10):
        v = [
             ut[11, :],
             ut[10, :],
             ut[9, :],
             ut[8, :]
            ]

        dv = np.zeros((4, 6, 3))

        for i in range(4):
            a, b = 0, 1
            for j in range(6):
                dv[i, j, 0] = v[i][3 * a] - v[i][3 * b]
                dv[i, j, 1] = v[i][3 * a + 1] - v[i][3 * b + 1]
                dv[i, j, 2] = v[i][3 * a + 2] - v[i][3 * b + 2]

                b += 1
                if b > 3:
                    a += 1
                    b = a + 1

        for i in range(6):

            l_6x10[i, 0] = self.dot(dv[0, i], dv[0, i])
            l_6x10[i, 1] = 2.0 * self.dot(dv[0, i], dv[1, i])
            l_6x10[i, 2] = self.dot(dv[1, i], dv[1, i])
            l_6x10[i, 3] = 2.0 * self.dot(dv[0, i], dv[2, i])
            l_6x10[i, 4] = 2.0 * self.dot(dv[1, i], dv[2, i])
            l_6x10[i, 5] = self.dot(dv[2, i], dv[2, i])
            l_6x10[i, 6] = 2.0 * self.dot(dv[0, i], dv[3, i])
            l_6x10[i, 7] = 2.0 * self.dot(dv[1, i], dv[3, i])
            l_6x10[i, 8] = 2.0 * self.dot(dv[2, i], dv[3, i])
            l_6x10[i, 9] = self.dot(dv[3, i], dv[3, i])

    def compute_rho(self, rho):
        rho[0] = self.dist2(self.cws[0], self.cws[1])
        rho[1] = self.dist2(self.cws[0], self.cws[2])
        rho[2] = self.dist2(self.cws[0], self.cws[3])
        rho[3] = self.dist2(self.cws[1], self.cws[2])
        rho[4] = self.dist2(self.cws[1], self.cws[3])
        rho[5] = self.dist2(self.cws[2], self.cws[3])

    def compute_a_and_b_gauss_newton(self, l_6x10, rho, betas, A, B):
        for i in range(6):
            rowL = l_6x10[i , :]
            A[i, 0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3]
            A[i, 1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3]
            A[i, 2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3]
            A[i, 3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3]

            B[i] = rho[i] - (
                rowL[0] * betas[0] ** 2 +
                rowL[1] * betas[0] * betas[1] +
                rowL[2] * betas[1] ** 2 +
                rowL[3] * betas[0] * betas[2] +
                rowL[4] * betas[1] * betas[2] +
                rowL[5] * betas[2] ** 2 +
                rowL[6] * betas[0] * betas[3] +
                rowL[7] * betas[1] * betas[3] +
                rowL[8] * betas[2] * betas[3] +
                rowL[9] * betas[3] ** 2
            )
        return A, B

    def gauss_newton(self, L_6x10, Rho, betas):
        iterations_number = 5

        A = np.zeros((6, 4))
        B = np.zeros(6)
        X = np.zeros(4)

        for _ in range(iterations_number):
            A, B = self.compute_a_and_b_gauss_newton(L_6x10, Rho, betas, A, B)
            X = np.linalg.solve(A.T @ A, A.T @ B)

            betas += X

        return betas

    def relative_error(rot_err, transl_err, Rtrue, ttrue, Rest, test):
        qtrue = PnPsolver.mat_to_quat(Rtrue)
        qest = PnPsolver.mat_to_quat(Rest)

        norm_qtrue = np.linalg.norm(qtrue)
        rot_err1 = np.linalg.norm(qtrue - qest) / norm_qtrue
        rot_err2 = np.linalg.norm(qtrue + qest) / norm_qtrue

        rot_err[0] = min(rot_err1, rot_err2)

        norm_ttrue = np.linalg.norm(ttrue)
        transl_err[0] = np.linalg.norm(np.array(ttrue) - np.array(test)) / norm_ttrue

    @staticmethod
    def mat_to_quat(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        q = np.zeros(4)

        if tr > 0.0:
            q[0] = R[1, 2] - R[2, 1]
            q[1] = R[2, 0] - R[0, 2]
            q[2] = R[0, 1] - R[1, 0]
            q[3] = tr + 1.0
            n4 = q[3]
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            q[0] = 1.0 + R[0, 0] - R[1, 1] - R[2, 2]
            q[1] = R[1, 0] + R[0, 1]
            q[2] = R[2, 0] + R[0, 2]
            q[3] = R[1, 2] - R[2, 1]
            n4 = q[0]
        elif R[1, 1] > R[2, 2]:
            q[0] = R[1, 0] + R[0, 1]
            q[1] = 1.0 + R[1, 1] - R[0, 0] - R[2, 2]
            q[2] = R[2, 1] + R[1, 2]
            q[3] = R[2, 0] - R[0, 2]
            n4 = q[1]
        else:
            q[0] = R[2, 0] + R[0, 2]
            q[1] = R[2, 1] + R[1, 2]
            q[2] = 1.0 + R[2, 2] - R[0, 0] - R[1, 1]
            q[3] = R[0, 1] - R[1, 0]
            n4 = q[2]

        scale = 0.5 / np.sqrt(n4)
        q *= scale
        return q

