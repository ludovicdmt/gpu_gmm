import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min


class GaussianMixture():

    def __init__(self, COMPONENTS, BATCH_SIZE, TRAINING_STEPS=1000, TOLERANCE = 1e-6, T0 = 500, alpha_eta = 0.6, verbose = 1):

        self.COMPONENTS = COMPONENTS
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAINING_STEPS = TRAINING_STEPS
        self.TOLERANCE = TOLERANCE
        self.T0 = T0
        self.alpha_eta = alpha_eta
        self.verbose = verbose
        # creating session
        self.sess = tf.InteractiveSession()


    def bic(self, ll, X):
        """Bayesian information criterion for the current model on the input X.
        Parameters
        ----------
        ll : maximum value of log likelihood
        X : array of shape (n_samples, n_dimensions)
        n_components : number of components in GMM
        Returns
        -------
        bic : float
            The lower the better.
        """
        return (-2 * ll * X.shape[0] +
                self.COMPONENTS * np.log(X.shape[0]))

    def aic(self, ll, X):
        """Akaike information criterion for the current model on the input X.
        Parameters
        ----------
        ll : maximum value of log likelihood
        X : array of shape (n_samples, n_dimensions)
        n_components : number of components in GMM
        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * ll * X.shape[0] + 2 * self.COMPONENTS


    def initialisation(self, data):

        self.DIMENSIONS = data.shape[1]
        # model inputs: data points (images)
        self.input = tf.placeholder(tf.float32, [None, self.DIMENSIONS])
        self.alpha = tf.placeholder_with_default(tf.cast(1.0, tf.float32), [])
        self.beta = tf.placeholder_with_default(tf.cast(1.0, tf.float32), [])
        print('Place holders defined')

        # constants: D*ln(2*pi), variance prior parameters
        self.ln2piD = tf.constant(np.log(2 * np.pi) * self.DIMENSIONS, dtype=tf.float32)

        # computing input statistics
        dim_means = tf.reduce_mean(self.input, 0)
        dim_distances = tf.squared_difference(self.input, tf.expand_dims(dim_means, 0))
        dim_variances = tf.reduce_sum(dim_distances, 0) / tf.cast(tf.shape(self.input)[0], tf.float32)
        avg_dim_variance = tf.cast(tf.reduce_sum(dim_variances) / self.COMPONENTS / self.DIMENSIONS, tf.float32)

        ## MiniBatchKmeans init

        mKmeans = MiniBatchKMeans(n_clusters=self.COMPONENTS, batch_size=self.BATCH_SIZE)
        mKmeans.fit(data)
        rand_point_ids, _ = pairwise_distances_argmin_min(mKmeans.cluster_centers_, data)  # Kmeans init
        # rand_point_ids = tf.squeeze(tf.multinomial(tf.ones([1, tf.shape(self.input)[0]]), COMPONENTS)) # Random init

        print('Input statistics computed')

        # default initial values of the variables
        initial_means = tf.placeholder_with_default(
            tf.gather(self.input, rand_point_ids),
            shape=[self.COMPONENTS, self.DIMENSIONS]
        )
        initial_covariances = tf.placeholder_with_default(
            tf.eye(self.DIMENSIONS, batch_shape=[self.COMPONENTS], dtype=tf.float32) * avg_dim_variance,
            shape=[self.COMPONENTS, self.DIMENSIONS, self.DIMENSIONS]
        )
        initial_weights = tf.placeholder_with_default(
            tf.cast(tf.constant(1.0 / self.COMPONENTS, shape=[self.COMPONENTS]), tf.float32),
            shape=[self.COMPONENTS]
        )

        # trainable variables: component means, covariances, and weights
        self.means = tf.Variable(initial_means, dtype=tf.float32)
        self.covariances = tf.Variable(initial_covariances, dtype=tf.float32)
        self.weights = tf.Variable(initial_weights, dtype=tf.float32)
        self.eta = tf.Variable((1 + self.T0) ** -self.alpha_eta, dtype=tf.float32)


    def training_step(self):

        # E-step: recomputing responsibilities with respect to the current parameter values
        differences = tf.subtract(tf.expand_dims(self.input, 0), tf.expand_dims(self.means, 1))
        diff_times_inv_cov = tf.matmul(differences, tf.matrix_inverse(self.covariances + 1e-6 * tf.eye(self.DIMENSIONS, batch_shape=[self.COMPONENTS], dtype=tf.float32)))
        sum_sq_dist_times_inv_cov = tf.reduce_sum(diff_times_inv_cov * differences, 2)
        # If batch_size is two small regarding DIMENSIONS size, covariances could be not inversible and so matrix_determinant is infinite
        log_coefficients = tf.expand_dims(self.ln2piD + tf.log(
            tf.matrix_determinant(self.covariances + 1e-6 * tf.eye(self.DIMENSIONS, batch_shape=[self.COMPONENTS], dtype=tf.float32))),
                                          1)
        log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)
        log_weighted = log_components + tf.expand_dims(tf.log(self.weights), 1)
        log_shift = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)
        exp_log_shifted = tf.exp(log_weighted - log_shift)
        exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, 0)
        gamma = exp_log_shifted / exp_log_shifted_sum
        gamma_sum = tf.reduce_sum(gamma, 1)
        gamma_weighted = gamma / tf.expand_dims(gamma_sum, 1)

        # M-step: maximizing parameter values with respect to the computed responsibilities
        means_ = tf.reduce_sum(tf.expand_dims(self.input, 0) * tf.expand_dims(gamma_weighted, 2), 1)
        differences_ = tf.subtract(tf.expand_dims(self.input, 0), tf.expand_dims(means_, 1))
        sq_dist_matrix = tf.matmul(tf.expand_dims(differences_, 3), tf.expand_dims(differences_, 2))
        covariances_ = tf.reduce_sum(sq_dist_matrix * tf.expand_dims(tf.expand_dims(gamma_weighted, 2), 3), 1)
        weights_ = gamma_sum / tf.cast(tf.shape(self.input)[0], tf.float32)

        # applying prior to the computed covariances
        covariances_ *= tf.expand_dims(tf.expand_dims(gamma_sum, 1), 2)
        covariances_ += tf.expand_dims(tf.diag(tf.fill([self.DIMENSIONS], 2.0 * self.beta)), 0)
        covariances_ /= tf.expand_dims(tf.expand_dims(gamma_sum + (2.0 * (self.alpha + 1.0)), 1), 2)

        # log-likelihood: objective function being maximized up to a TOLERANCE delta
        log_likelihood = tf.reduce_sum(tf.log(exp_log_shifted_sum)) + tf.reduce_sum(log_shift)
        self.mean_log_likelihood = log_likelihood / tf.cast(tf.shape(self.input)[0] * tf.shape(self.input)[1], tf.float32)

        # updating the parameters by new values
        train_step = tf.group(
            self.means.assign(self.means * (1 - self.eta) + self.eta * means_),
            self.covariances.assign(self.covariances * (1 - self.eta) + self.eta * covariances_),
            self.weights.assign(self.weights * (1 - self.eta) + self.eta * weights_)
        )

        return train_step

    def fit(self, data):

        # Init step
        self.initialisation(data=data)

        train_step = self.training_step()

        # initializing trainable variables
        self.sess.run(tf.global_variables_initializer(), feed_dict={self.input: data})

        previous_likelihood = -np.inf

        # training loop
        for step in range(self.TRAINING_STEPS):
            if step < 10:
                self.eta.assign(self.eta / 2)
            else:
                self.eta.assign((step + self.T0) ** - self.alpha_eta)

            # executing a training step and
            # fetching evaluation information

            # Online (incremental EM)
            batch_idx = np.random.choice(range(len(data)), size=self.BATCH_SIZE, replace=False)
            batch = data[batch_idx]

            current_likelihood, _ = self.sess.run(
                [self.mean_log_likelihood, train_step],
                feed_dict={self.input: batch}
            )
            if step > 0:
                # computing difference between consecutive likelihoods
                difference = np.abs(current_likelihood - previous_likelihood)
                if self.verbose == 1:
                    print("{0}:\tmean-likelihood {1:.8f}\tdifference {2}".format(
                        step, current_likelihood, difference))

                # stopping if TOLERANCE reached
                if difference <= self.TOLERANCE:
                    break
            else:
                if self.verbose == 1:
                    print("{0}:\tmean-likelihood {1:.8f}".format(
                        step, current_likelihood))

            previous_likelihood = current_likelihood


    def predict_proba(self, data):
        ll, weights = self.sess.run(
            [self.log_likelihood, self.gamma_weighted],
            feed_dict={self.input: data}
        )

        return weights




