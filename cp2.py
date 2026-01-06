import pickle
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac


def parafac2_als(X_list, R, max_iter=50, tol=1e-6):
    K = len(X_list)
    J = X_list[0].shape[1]

    H = np.random.randn(R, R)
    V = np.random.randn(J, R)

    S_list = [np.eye(R) for _ in range(K)]
    Q_list = [None] * K 

    prev_obj = np.inf
    for it in range(max_iter):
        for k in range(K):
            Xk = X_list[k]  # (Iₖ, J)
            M_k = H @ S_list[k] @ (V.T @ Xk.T)
            U, sigma, Vt = np.linalg.svd(M_k, full_matrices=False)
            Qk = (Vt.T) @ U.T
            Q_list[k] = Qk

        Y_tensor = np.zeros((R, J, K))
        for k in range(K):
            Xk = X_list[k]
            Qk = Q_list[k]
            Yk = Qk.T @ Xk  # (R, J)
            Y_tensor[:, :, k] = Yk


        W_init = np.random.randn(K, R)
        factors_init = [H, V, W_init]
        weights_init = np.ones(R)  
        cp_init = (weights_init, factors_init)
        cp_decomp = parafac(Y_tensor, rank=R, n_iter_max=1, init=cp_init, tol=tol, verbose=False)
        H_new, V_new, W = cp_decomp.factors  # H_new: (R×R), V_new: (J×R), W: (K×R)
        H = H_new
        V = V_new

        for k in range(K):
            S_list[k] = np.diag(W[k, :])

        obj = 0.0
        for k in range(K):
            Xk = X_list[k]
            Qk = Q_list[k]
            Sk = S_list[k]
            Xk_hat = Qk @ H @ Sk @ V.T
            obj += np.linalg.norm(Xk - Xk_hat) ** 2
        obj = np.sqrt(obj)

        if abs(prev_obj - obj) < tol * prev_obj:
            print(f"Converged at iteration {it + 1} with objective {obj:.6f}")
            break
        prev_obj = obj
        print(f"Iteration {it + 1}, objective: {obj:.6f}")

    U_list = []
    Xhat_list = []
    for k in range(K):
        Qk = Q_list[k]
        Sk = S_list[k]
        Uk = Qk @ H  # (Iₖ, R)
        U_list.append(Uk)
        Xk_hat = Uk @ Sk @ V.T  # (Iₖ, J)
        Xhat_list.append(Xk_hat)

    norm_X = 0.0
    norm_res = 0.0
    for k in range(K):
        norm_X += np.linalg.norm(X_list[k]) ** 2
        norm_res += np.linalg.norm(X_list[k] - Xhat_list[k]) ** 2
    norm_X = np.sqrt(norm_X)
    norm_res = np.sqrt(norm_res)
    fit = 1 - (norm_res / norm_X)

    return Xhat_list, fit, U_list, S_list, V, Q_list


if __name__ == "__main__":
    with open("assistant_embedding_matrices_Math_gpt4o.pkl", "rb") as f:
        assistant_embedding_matrix_dict = pickle.load(f)
    with open("user_embedding_matrices_Math_gpt4o.pkl", "rb") as f:
        user_embedding_matrix_dict = pickle.load(f)
    fit_dict = {}

    for key in user_embedding_matrix_dict.keys():
        print(f"\nProcessing key: {key}")
        user_embedding_matrix_list = user_embedding_matrix_dict[key]
        assistant_embedding_matrix_list = assistant_embedding_matrix_dict[key]
        combined_list = []
        for i in range(10):
            combined_list.append(user_embedding_matrix_list[i])
            combined_list.append(assistant_embedding_matrix_list[i])

        normalized_list = []
        for mat in combined_list:
            mat = np.array(mat, dtype=float)
            norm = np.linalg.norm(mat)
            if norm > 0:
                mat = mat / norm
            normalized_list.append(mat)
        combined_list = normalized_list

        fit_list = []
        for target_rank in range(1, 51):
            print(f"\nRunning PARAFAC2-ALS with target rank {target_rank} for key {key} ...")
            try:
                _, fit, _, _, _, _ = parafac2_als(combined_list, target_rank, max_iter=50, tol=1e-6)
            except:
                fit = 0
            print(f"Key {key}, target rank {target_rank}: fit = {fit:.4f}")
            fit_list.append(fit)

        fit_dict[key] = fit_list

    with open("fit_dict_Math_camel_gpt4o.pkl", "wb") as f:
        pickle.dump(fit_dict, f)

    print("\nCP2 Done")