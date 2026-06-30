import numpy as np


def perturb(org_data, mechanism, scale, public_data=None):
    if mechanism not in ['none', 'gauss', 'lap', 'cgm']:
        raise RuntimeError('Unsupported mechanism')
    elif mechanism == 'gauss':
        return gaussian_mechanism(org_data, scale)
    elif mechanism == 'lap':
        return laplace_mechanism(org_data, scale)
    elif mechanism == 'cgm':
        return corr_gaussian_mechanism(org_data, scale, public_data)
    elif mechanism == 'none':
        return org_data


def gaussian_mechanism(org_data, scale):
    record_count = org_data.shape[0]
    dim_count = org_data.shape[1]

    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        noise = np.random.normal(0, scale, (record_count, dim_count))
        return org_data + noise
    elif isinstance(scale, list):
        scale = np.array(scale)
    elif isinstance(scale, np.ndarray):
        scale = scale
    else:
        raise RuntimeError('Unsupported type of σ')
    if scale.shape[0] != dim_count:
        raise RuntimeError('Incorrect shape of σ')

    noise = []
    for i in range(dim_count):
        noise.append(np.random.normal(0, scale[i], record_count))
    noise = np.array(noise)
    noise = noise.transpose()

    return org_data + noise


def laplace_mechanism(org_data, scale):
    record_count = org_data.shape[0]
    dim_count = org_data.shape[1]

    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        noise = np.random.laplace(0, scale, (record_count, dim_count))
        return org_data + noise
    elif isinstance(scale, list):
        scale = np.array(scale)
    elif isinstance(scale, np.ndarray):
        scale = scale
    else:
        raise RuntimeError('Unsupported type of σ')
    if scale.shape[0] != dim_count:
        raise RuntimeError('Incorrect shape of σ')

    noise = []
    for i in range(dim_count):
        noise.append(np.random.laplace(0, scale[i], record_count))
    noise = np.array(noise)
    noise = noise.transpose()

    return org_data + noise


def corr_gaussian_mechanism(org_data, scale, public_data):
    record_count = org_data.shape[0]
    dim_count = org_data.shape[1]
    
    if public_data is None:
        raise RuntimeError('D_pub cannot be null')

    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        scale = np.ones(dim_count) * scale
    elif isinstance(scale, list):
        scale = np.array(scale)
    elif isinstance(scale, np.ndarray):
        scale = scale
    else:
        raise RuntimeError('Unsupported type of σ')
    if scale.shape[0] != dim_count:
        raise RuntimeError('Incorrect shape of σ')

    corr = np.corrcoef(public_data, rowvar=False)

    mean = np.zeros(dim_count)
    sigma = np.diag(scale)
    cov = sigma @ corr @ sigma
    # Ensure positive definiteness
    cov = (cov + cov.T) / 2

    noise = np.random.multivariate_normal(mean, cov, record_count)

    return org_data + noise


def corr_gaussian_2d(org_data, scale, rho):
    if rho >= 1 or rho <= -1:
        raise RuntimeError('ρ should be less than 1 and greater than -1')

    record_count = org_data.shape[0]
    dim_count = org_data.shape[1]

    if dim_count != 2:
        raise RuntimeError('2-D data only')

    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        scale = np.ones(dim_count) * scale
    elif isinstance(scale, list):
        scale = np.array(scale)
    elif isinstance(scale, np.ndarray):
        scale = scale
    else:
        raise RuntimeError('Unsupported type of σ')
    if scale.shape[0] != dim_count:
        raise RuntimeError('Incorrect shape of σ')

    corr = np.array([[1, rho], [rho, 1]])
    mean = np.zeros(dim_count)
    sigma = np.diag(scale)
    cov = sigma @ corr @ sigma
    # Ensure positive definiteness
    cov = (cov + cov.T) / 2

    noise = np.random.multivariate_normal(mean, cov, record_count)

    return org_data + noise
