import numpy as np
from sklearn.linear_model import LinearRegression


def psd_project(A: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Project a symmetric matrix A onto the PSD cone by eigenvalue clipping.
    eps: minimum eigenvalue after clipping (use small >0 for numerical stability).
    """
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w_clipped = np.maximum(w, eps)
    return (V * w_clipped) @ V.T


def shrink_cov(S: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Simple shrinkage: (1-alpha)*S + alpha*tau*I, tau = tr(S)/d.
    alpha=0 means no shrinkage.
    """
    if alpha <= 0:
        return S
    d = S.shape[0]
    tau = np.trace(S) / d
    return (1 - alpha) * S + alpha * tau * np.eye(d)


def estimate_linear_relation_from_noisy(
        x_tilde: np.ndarray,
        y_tilde: np.ndarray,
        sigma2: float,
        eps: float = 1e-5,
        shrink_alpha: float = 0.05,
):
    """
    Estimate rho, k, b for Y ≈ kX + b using only noisy observations:
        x_tilde = x + n_x, y_tilde = y + n_y
    with known noise variance sigma2 (assumed same on both dims here).

    Uses:
      1) sample covariance of [x_tilde, y_tilde]
      2) optional shrinkage on that covariance
      3) subtract sigma2*I
      4) PSD projection to avoid negative variances
      5) compute rho, k, b
    """
    x_tilde = np.asarray(x_tilde).ravel()
    y_tilde = np.asarray(y_tilde).ravel()
    assert x_tilde.shape == y_tilde.shape
    n = x_tilde.size

    # sample means (noise is zero-mean => means unchanged in expectation)
    mx = x_tilde.mean()
    my = y_tilde.mean()

    # sample covariance (unbiased, ddof=1)
    Z = np.vstack([x_tilde, y_tilde])
    S_tilde = np.cov(Z, ddof=1)

    # optional shrinkage for stability
    S_tilde = shrink_cov(S_tilde, alpha=shrink_alpha)

    # de-noise
    A = S_tilde - sigma2 * np.eye(2)

    # choose eps if not provided
    if eps is None:
        # scale-aware tiny floor for numerical stability
        eps = 1e-12 * max(np.trace(S_tilde), 1.0)

    # PSD projection
    S_hat = psd_project(A, eps=eps)

    vX = S_hat[0, 0]
    vY = S_hat[1, 1]
    cXY = S_hat[0, 1]

    # rho, k, b
    rho = cXY / np.sqrt(vX * vY)
    k = cXY / vX
    b = my - k * mx

    return float(rho), float(k), float(b)


def reconstruction(noisy_data, public_data, threshold=0.75):
    record_count = noisy_data.shape[0]
    dim_count = noisy_data.shape[1]

    corr = np.corrcoef(public_data, rowvar=False)
    final_result = []
    # Reconstruct dimension i
    for i in range(dim_count):
        noisy_i = noisy_data[:, i]
        infer_i = []
        for j in range(dim_count):
            if i == j:
                continue
            corr_ij = corr[i][j]
            # Find highly correlated dimensions
            if (threshold <= corr_ij <= 1.0) or (-1.0 <= corr_ij <= -threshold):
                noisy_j = noisy_data[:, j]

                # Linear Regression
                model = LinearRegression()
                X = public_data[:, j].reshape(-1, 1)
                Y = public_data[:, i].reshape(-1)
                model.fit(X, Y)
                alpha = model.coef_[0]
                beta = model.intercept_
                # ''' noisy version '''
                # corr_ij, alpha, beta = estimate_linear_relation_from_noisy(noisy_data[:, j], noisy_data[:, i], 1)

                infer_i_by_j = alpha * noisy_j + beta
                infer_i.append((corr_ij, infer_i_by_j))
        final_i = np.zeros(record_count) + noisy_i
        weights_i = 1
        # Weighted average
        for k in range(len(infer_i)):
            final_i += infer_i[k][0] * infer_i[k][0] * infer_i[k][1]
            weights_i += infer_i[k][0] * infer_i[k][0]
            # final_i += infer_i[k][1]
            # weights_i += 1
        final_i /= weights_i
        final_result.append(final_i)
    final_result = np.array(final_result).transpose(1, 0)
    return final_result
