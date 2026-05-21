import numpy as np


def result_analyze(reconstruction_result, noisy_data, org_data):
    dim_count = org_data.shape[1]

    mse_ptb = np.mean(np.sum((noisy_data - org_data) ** 2, axis=1) / dim_count)
    mse_rec = np.mean(np.sum((reconstruction_result - org_data) ** 2, axis=1) / dim_count)
    rg = (1 - (mse_rec / mse_ptb)) * 100

    return mse_ptb, mse_rec, rg
