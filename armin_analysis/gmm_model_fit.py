import numpy as np

def genGaussian(x, mu, sigma):
    value = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (((x - mu) / sigma) ** 2))

    return value

def genMix(x, k, w, mu, sigma):
    value = np.zeros((len(x)), dtype=float)

    for i in range(k):
        value += w[i] * genGaussian(x, mu[i], sigma[i])

    # normalize
    value = value / value.sum()

    return value


def init_params(k, variant, experiment='dot_motion'):
    w = np.ones(k) / k
    m = np.deg2rad([-40.0, 0.0, 40.0])

    if variant == 'fixed_control' or variant == 'fixed_s':
        if experiment == 'dot_motion_fb':
            s = np.deg2rad([23.23, 4.59, 23.23])
        elif experiment == 'dot_motion':
            s = np.deg2rad([25.8, 4.27, 25.8])
        else:
            s = np.deg2rad([21.0, 4.7, 21.0])
    else:
        s = np.deg2rad([20.0, 5.0, 20.0])

    return w, m, s


def EM(X, k, w, mu, sigma, variant, same=False):
    max_iter = 50
    epsilon = 1e-7

    for iteration in range(max_iter):

        cluster_pdf = np.zeros((k, X.shape[0]))

        for cluster in range(k):
            cluster_pdf[cluster] = genGaussian(X, mu[cluster], sigma[cluster])

        weighted_pdf = np.multiply(cluster_pdf, np.tile(w, (X.shape[0], 1)).T)
        gamma = weighted_pdf / (np.sum(weighted_pdf, axis=0) + epsilon)

        w = np.mean(gamma, axis=1)

        gamma_sum = np.sum(gamma, axis=1) + epsilon

        if variant in ['fixed_mcs', 'fixed_m', 'fixed_initial']:
            sigma = np.sqrt(np.diagonal(np.dot(gamma, np.square(np.tile(X, (k, 1)).T - mu))) / gamma_sum)
        if variant in ['fixed_initial', 'fixed_s', 'fixed_ncm']:
            mu = np.dot(gamma, X) / gamma_sum
        if variant == 'fixed_ncm':
            mu[0] = np.deg2rad(-40.0)
            mu[2] = np.deg2rad(40.0)

        if same:
            w[0] = w[2]
            if variant in ['fixed_mcs', 'fixed_m', 'fixed_initial']:
                sigma[0] = sigma[2]

            if variant == 'fixed_initial' or variant == 'fixed_s':
                mu[0] = mu[2]

        if variant == 'fixed_mcs':
            sigma[1] = np.deg2rad(5.0)

    w = np.around(w, 3)
    diff = 1 - sum(w)

    while (diff != 0):
        if diff < 0:
            w += diff / k
        elif diff < 1e-3:
            w[1] += diff
        else:
            w += diff / k
        diff = 1 - sum(w)

    return w, mu, sigma


def gmm_model_fit(bins, probabilities):  # your histogram values are in data

    #variants = ['fixed_initial', 'fixed_m', 'fixed_mcs', 'fixed_ms', 'fixed_control']

    # fixed_initial = all parameters can vary
    # fixed_m = mean stays constant
    # fixed_mcs = mean and variance of center are constant
    # fixed_ms = mean and variance are fixed
    # fixed_control = mean and variance are fixed based on control condition

    variant = 'fixed_initial'

    k = 3  # number of clusters
    num_points = 5000
    samples = np.random.choice(np.radians(bins), num_points, p=probabilities)

    w, m, s = init_params(k, variant=variant)  # initialize

    # # Decide if center mean should be zero or shifted
    # if abs(data[num_bins // 2] - data[num_bins // 2 - 1]) < 1e-3:
    #     m[1] = 0.0
    # else:
    #     m[1] = angles[np.argmax(data)]

    m[1] = 0.0

    fit_w, fit_m, fit_s = EM(samples, k, w, m, s, variant)  # fit model

    return fit_w, np.degrees(fit_m), np.degrees(fit_s)