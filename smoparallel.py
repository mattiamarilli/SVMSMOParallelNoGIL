from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
import numpy as np
from threading import Thread
import time
import math



class SVM:
    def __init__(
        self,
        numthreads: int = 1,
        c: float = 1.,
        kkt_thr: float = 1e-3,
        max_iter: int = 1e4,
        kernel_type: str = 'linear',
        gamma_rbf: float = 1.
    ) -> None:
        if kernel_type not in ['linear', 'rbf']:
            raise ValueError('kernel_type must be either {} or {}'.format('linear', 'rbf'))
        super().__init__()
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.kkt_thr = kkt_thr
        self.gamma_rbf = gamma_rbf
        self.b = 0.0
        self.alpha = np.array([])
        self.support_vectors = np.array([])
        self.support_labels = np.array([])
        self.numthreads = numthreads
        self.sum_columns_calculation_time = 0
        self.thread_pool = ThreadPoolExecutor(max_workers=numthreads)


        if kernel_type == 'linear':
            self.kernel = self.linear_kernel
        elif kernel_type == 'rbf':
            self.kernel = self.rbf_kernel
            self.gamma_rbf = gamma_rbf

    # ------------------- PREDICT -------------------
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.alpha.shape[0] == 0:
            raise ValueError("Model not trained yet.")
        K_test = self.kernel(self.support_vectors, x)
        scores = np.dot(self.alpha * self.support_labels, K_test) + self.b
        pred = np.sign(scores)
        return pred, scores

    def mvp_selection(self, error_cache: np.ndarray) -> Tuple[int, int]:
        """
        Select the pair of alphas that most violate the KKT conditions.

        Args:
            error_cache (np.ndarray): Current error cache.

        Returns:
            Tuple[int, int]: Indices of the selected alphas (i, j).
        """
        alpha = self.alpha
        y = self.support_labels
        C = self.c

        # Determine sets L, U, Free
        L_indices = (alpha <= 1e-10)
        U_indices = (alpha >= C - 1e-10)
        Free_indices = (alpha > 1e-10) & (alpha < C - 1e-10)

        # Create masks for R and S sets
        R_mask = (L_indices & (y == 1)) | (U_indices & (y == -1)) | Free_indices
        S_mask = (L_indices & (y == -1)) | (U_indices & (y == 1)) | Free_indices

        R_indices = np.where(R_mask)[0]
        S_indices = np.where(S_mask)[0]

        # Remove NaN values from consideration
        R_indices = [i for i in R_indices if not np.isnan(error_cache[i])]
        S_indices = [i for i in S_indices if not np.isnan(error_cache[i])]

        if len(R_indices) == 0 or len(S_indices) == 0:
            return -1, -1

        # Check maximal KKT violation
        max_violation = np.max(error_cache[S_indices]) - np.min(error_cache[R_indices])
        if max_violation < self.kkt_thr:
            return -1, -1

        # Select indices with maximal violation
        i_mvp = R_indices[np.argmin(error_cache[R_indices])]
        j_mvp = S_indices[np.argmax(error_cache[S_indices])]

        return i_mvp, j_mvp

    # ------------------- FIT -------------------
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        N, D = x_train.shape
        self.b = 0.0
        self.alpha = np.zeros(N)
        self.support_labels = y_train
        self.support_vectors = x_train
        iter_idx = 0

        # calcolo iniziale cache errori
        if self.kernel == self.linear_kernel:
            K = x_train @ x_train.T
            error_cache = (self.alpha * y_train) @ K + self.b - y_train
        elif self.kernel == self.rbf_kernel:
            sq_norms = np.sum(x_train ** 2, axis=1)
            K = np.exp(
                -self.gamma_rbf *
                (sq_norms[:, None] + sq_norms[None, :] - 2 * x_train @ x_train.T)
            )
            error_cache = (self.alpha * y_train) @ K + self.b - y_train

        print("SVM training using SMO algorithm - START")

        while iter_idx < self.max_iter:
            i_2, i_1 = self.mvp_selection(error_cache)

            # Seleziona i_MVP e j_MVP
            if i_2 == -1 or i_1 == -1:
                break

            y_1, alpha_1 = self.support_labels[i_1], self.alpha[i_1]
            y_2, alpha_2 = self.support_labels[i_2], self.alpha[i_2]

            # --- Precalcolo colonne kernel ---
            start_time = time.time()
            K_i1 = np.array(self.rbf_kernel_column_multithread(i_1, gamma_rbf=self.gamma_rbf))
            K_i2 = np.array(self.rbf_kernel_column_multithread(i_2, gamma_rbf=self.gamma_rbf))
            #print(f"Parallelo: {(time.time() - start_time)}")
            self.sum_columns_calculation_time += time.time() - start_time

            k11 = K_i1[i_1]
            k22 = K_i2[i_2]
            k12 = K_i1[i_2]

            # --- Calcolo boundaries ---
            L, H = self.compute_boundaries(alpha_1, alpha_2, y_1, y_2)

            # --- Calcolo eta ---
            eta = k11 + k22 - 2 * k12
            if eta < 1e-12:
                continue

            # --- Calcolo errori ---
            E_1 = np.dot(self.alpha * self.support_labels, K_i1) + self.b - y_1
            E_2 = np.dot(self.alpha * self.support_labels, K_i2) + self.b - y_2

            # --- Aggiornamento alpha ---
            alpha_2_new = alpha_2 + y_2 * (E_1 - E_2) / eta
            alpha_2_new = np.clip(alpha_2_new, L, H)
            alpha_1_new = alpha_1 + y_1 * y_2 * (alpha_2 - alpha_2_new)

            # --- Aggiornamento b ---
            b1 = (
                self.b - E_1
                - y_1 * (alpha_1_new - alpha_1) * k11
                - y_2 * (alpha_2_new - alpha_2) * k12
            )
            b2 = (
                self.b - E_2
                - y_1 * (alpha_1_new - alpha_1) * k12
                - y_2 * (alpha_2_new - alpha_2) * k22
            )

            if 0 < alpha_1_new < self.c:
                self.b = b1
            elif 0 < alpha_2_new < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2

            # --- Salvataggio alpha aggiornati ---
            self.alpha[i_1] = alpha_1_new
            self.alpha[i_2] = alpha_2_new

            # --- Aggiornamento cache errori ---
            delta_alpha_1 = alpha_1_new - alpha_1
            delta_alpha_2 = alpha_2_new - alpha_2
            error_cache += y_1 * delta_alpha_1 * K_i1 + y_2 * delta_alpha_2 * K_i2

            iter_idx += 1

        # --- Filtraggio support vectors ---
        end_time_fit = time.time()
        support_vectors_idx = (self.alpha != 0)
        self.support_labels = self.support_labels[support_vectors_idx]
        self.support_vectors = self.support_vectors[support_vectors_idx, :]
        self.alpha = self.alpha[support_vectors_idx]

        print(f"Training summary: {iter_idx} iterations")
        print(f"Tempo calcolo colonne: {self.sum_columns_calculation_time}")
        print("SVM training using SMO algorithm - DONE!")

    # ------------------- BOUNDS -------------------
    def compute_boundaries(self, alpha_1, alpha_2, y_1, y_2) -> Tuple[float, float]:
        if y_1 == y_2:
            lb = max(0, alpha_1 + alpha_2 - self.c)
            ub = min(self.c, alpha_1 + alpha_2)
        else:
            lb = max(0, alpha_2 - alpha_1)
            ub = min(self.c, self.c + alpha_2 - alpha_1)
        return lb, ub

    def rbf_kernel_column_multithread(self, i: int, gamma_rbf: float):
        support_vectors = self.support_vectors
        n = len(support_vectors)
        col = [0.0] * n

        def worker(start, end):
            for idx in range(start, end):
                sq_norm = 0.0
                for d1, d2 in zip(support_vectors[idx], support_vectors[i]):
                    delta = d1 - d2
                    sq_norm += delta * delta
                col[idx] = math.exp(-gamma_rbf * sq_norm)

        batch_size = n // self.numthreads
        futures = []
        for t in range(self.numthreads):
            start_idx = t * batch_size
            end_idx = n if t == self.numthreads - 1 else (t + 1) * batch_size
            futures.append(self.thread_pool.submit(worker, start_idx, end_idx))

        for f in futures:
            f.result()

        return col

    # ------------------- KERNELS -------------------
    def rbf_kernel(self, u, v):
        if np.ndim(v) == 1:
            v = v[np.newaxis, :]
        if np.ndim(u) == 1:
            u = u[np.newaxis, :]
        dist_squared = np.linalg.norm(u[:, :, np.newaxis] - v.T[np.newaxis, :, :], axis=1) ** 2
        dist_squared = np.squeeze(dist_squared)
        return np.exp(-self.gamma_rbf * dist_squared)

    @staticmethod
    def linear_kernel(u, v) -> np.ndarray:
        return np.dot(u, v.T)


def compute_rbf_block(support_vectors, x_i, gamma_rbf, indices, result_list, offset):
    for idx, global_idx in enumerate(indices):
        diff = support_vectors[global_idx] - x_i
        result_list[offset + idx] = np.exp(-gamma_rbf * np.dot(diff, diff))


