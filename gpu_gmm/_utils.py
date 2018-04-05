import numpy as np



def generate_gmm_data(points, components, dimensions, seed):
    """Generates synthetic data of a given size from a random GMM"""
    np.random.seed(seed)

    c_means = np.random.normal(size=[components, dimensions]) * 10
    c_variances = np.abs(np.random.normal(size=[components, dimensions]))
    c_weights = np.abs(np.random.normal(size=[components]))
    c_weights /= np.sum(c_weights)

    result = np.zeros((points, dimensions), dtype=np.float32)

    for i in xrange(points):
        if i % 10000 ==0:
            print('Iteration ', i, ' on ', points)
        comp = np.random.choice(np.array(range(components)), p=c_weights)
        result[i] = np.random.multivariate_normal(
            c_means[comp], np.diag(c_variances[comp])
        )

    np.random.seed()

    return result, c_means, c_variances, c_weights


