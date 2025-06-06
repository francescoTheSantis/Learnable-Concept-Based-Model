import numpy as np
from sklearn.preprocessing import OneHotEncoder

nats2bits = 1.0 / np.log(2)


# noise_variance = 0.01


def get_dists(X: np.ndarray) -> np.ndarray:
    """Torch code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = np.expand_dims(np.sum(np.square(X), axis=1), 1)
    dists = x2 + x2.T - 2 * np.dot(X, X.T)  # dot in numpy with two matrices is equivalent to np.mm
    return dists


def get_shape(x):
    N = x.shape[1]
    P = x.shape[0]
    return N, P


def entropy_estimator_kl(x, sigma):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    N, P = get_shape(x)
    dists = get_dists(x)
    dists2 = dists / (2 * sigma)
    normconst = (N / 2.0) * np.log(2 * np.pi * sigma)
    lprobs = np.log(np.sum(np.exp(-dists2), axis=1)) - np.log(P) - normconst
    h = -np.mean(lprobs)
    assert not np.isinf(h), "Error while computing H"
    h_2 = N / 2 + h

    # return h
    return h_2


def entropy_estimator_bd(x, sigma):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    N, P = get_shape(x)
    val = entropy_estimator_kl(x, 4 * sigma)
    return val + np.log(0.25) * N / 2


def kde_condentropy(output, sigma):
    # Return entropy of a multivariate Gaussian, in nats
    N = output.shape[1]
    cond_ent = (N / 2.0) * (np.log(2 * np.pi * sigma) + 1)
    # assert cond_ent > 0, "Error while computing H(T|X)"
    return cond_ent


def mutual_information_X_T(output, entropy_T=None, normalized=False, return_H=False, noise_variance=0.01):

    if isinstance(entropy_T, type(None)):
        entropy_T = entropy_estimator_kl(output, noise_variance)
        entropy_T = entropy_T * nats2bits

    cond_entropyT_X = kde_condentropy(output, noise_variance)
    cond_entropyT_X = cond_entropyT_X * nats2bits

    mutual_info_X_T = entropy_T - cond_entropyT_X
    #assert mutual_info_X_T > -0.001, f"Error while computing I(X;T): {mutual_info_X_T}"

    if normalized:
        raise NotImplementedError()
        # mutual_info_X_T = mutual_info_X_T / np.max((np.abs(entropy_T), np.abs(cond_entropyT_X)))
        # assert 0 <= mutual_info_X_T, f"Error while normalizing I(X;T): {mutual_info_X_T}"

    if return_H:
        return mutual_info_X_T, entropy_T, cond_entropyT_X

    return mutual_info_X_T


def mutual_information_T_Y(output, labels, entropy_T=None, normalized=False, return_H=False, noise_variance=0.01):

    # print(labels.shape)
    # convert labels to one_hot encoding
    if len(labels.shape) == 1:
        labels = OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()

    # print(labels.shape)
    n_classes = labels.shape[1]
    label_probs = np.mean(labels, axis=0)
    assert np.abs(np.sum(label_probs) - 1) < 0.001, "Error in calculating label probabilities"

    cond_entropy_T_Y = .0
    for i in range(n_classes):
        # samples_class_i = labels == i
        samples_class_i = labels[:, i] != 0  # already boolean since we have multi-label labels
        output_class_i = output[samples_class_i, :]
        cond_entropy_i = label_probs[i] * entropy_estimator_kl(output_class_i, noise_variance)
        cond_entropy_T_Y += cond_entropy_i
        # print(f"{i}/{n_classes} {cond_entropy_i}")
    cond_entropy_T_Y = cond_entropy_T_Y * nats2bits

    if isinstance(entropy_T, type(None)):
        entropy_T = entropy_estimator_kl(output, noise_variance)
        entropy_T = entropy_T * nats2bits

    mutual_info_Y_T = entropy_T - cond_entropy_T_Y
    '''
    assert mutual_info_Y_T > -0.001, f"Error while computing I(T;Y): 1 >= {mutual_info_Y_T} > -.0001," \
                                          f"\nEntropy(T): {entropy_T}" \
                                          f"\nEntropy(T|Y): {cond_entropy_T_Y}" \
                                          f"\nLabels: {labels.shape} max: {labels.max()} min: {labels.min()}" \
                                          f"\nOutput: {output.shape} max: {output.max()} min: {output.min()}"
    '''
    if normalized:
        raise NotImplementedError()
        # mutual_info_Y_T = mutual_info_Y_T / np.max((np.abs(entropy_T), np.abs(mutual_info_Y_T)))
        # assert 0 <= mutual_info_Y_T <= 1, f"Error while normalizing I(X;T): {mutual_info_Y_T}"

    if return_H:
        return mutual_info_Y_T, entropy_T, cond_entropy_T_Y

    return mutual_info_Y_T


def mutual_information_T_C(output, labels, entropy_T=None, normalized=False, return_H=False, noise_variance=0.01):

    # convert labels to positive vs negative labels encoding
    if len(labels.squeeze().shape) > 1:
        n_classes = labels.shape[1]
    else:
        n_classes = np.unique(labels).shape[0]

    if isinstance(entropy_T, type(None)):
        entropy_T = entropy_estimator_kl(output, noise_variance)
        entropy_T = entropy_T * nats2bits

    cond_entropy_T_C = .0
    for i in range(n_classes):
        label_probs = np.mean(labels[:, i], axis=0)
        label_probs = [1 - label_probs, label_probs]
        assert np.abs(np.sum(label_probs) - 1) < 0.001, "Error in calculating label probabilities"
        for j in range(2):
            samples_class_i_j = labels[:, i] == j  # already boolean since we have multi-label labels
            output_class_i_j = output[samples_class_i_j, :]
            cond_entropy_i_j = label_probs[j] * entropy_estimator_kl(output_class_i_j, noise_variance)
            # cond_entropy_T_C += nats2bits * cond_entropy_i_j / 2  # taking into account negative and positive classes
            cond_entropy_T_C += nats2bits * cond_entropy_i_j
    cond_entropy_T_C /= n_classes  # averaging the measure over all the classes

    mutual_info_T_C = entropy_T - cond_entropy_T_C

    assert 1.1 >= mutual_info_T_C > -0.001, f"Error while computing I(T;C): {mutual_info_T_C}"

    if normalized:
        raise NotImplementedError()
        # mutual_info_T_C = mutual_info_T_C / np.max((np.abs(entropy_T), np.abs(mutual_info_T_C)))
        # assert 0 <= mutual_info_T_C <= 1, f"Error while normalizing I(X;T): {mutual_info_T_C}"

    if return_H:
        return mutual_info_T_C, entropy_T, cond_entropy_T_C

    return mutual_info_T_C

