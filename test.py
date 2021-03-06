from sklearn.datasets import make_blobs
from gpu_gmm import GaussianMixture
import numpy as np
from gpu_gmm import generate_gmm_data
import matplotlib.pyplot as plt


DIMENSIONS = 40
COMPONENTS = 5
DATA_POINTS = 500000

BATCH_SIZE = 10000
TRAINING_STEPS = 1000
TOLERANCE = 1e-6

T0 = 500
alpha_eta = 0.6

# Generating data
data, y = make_blobs(n_samples=DATA_POINTS, n_features=DIMENSIONS, centers=COMPONENTS,
                                   random_state=10)
#data, true_means, true_variances, true_weights = generate_gmm_data( DATA_POINTS, COMPONENTS, DIMENSIONS, 10)


gmm = GaussianMixture(COMPONENTS=COMPONENTS, BATCH_SIZE = BATCH_SIZE, verbose=2)

gmm.fit(data)


test_idx = np.random.choice(range(len(data)), size=5000, replace=False)
test = data[test_idx]

ll, features = gmm.predict_proba(test)
print(ll)
print(gmm.mean_log_likelihood)
print(features.shape)