import numpy as np
import tensorflow as tf
from sklearn import mixture

class ConvergenceException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


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
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.InteractiveSession()


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
        self.input = data
        self.alpha = 1.0
        self.beta = 1.0

        if self.verbose > 1:
            print('Place holders defined')

        # constants: D*ln(2*pi), variance prior parameters
        self.ln2piD = tf.constant(np.log(2 * np.pi) * self.DIMENSIONS, dtype=tf.float32)


        ## Init with GMM from sklearn with one full covariance matrix for all components (tied)
        gmm = mixture.GaussianMixture(n_components=self.COMPONENTS, covariance_type='full', max_iter=5)

        if len(data) > 50000:  # Only work on a subset to accelerate computing
            gmm.fit(data[np.random.choice(len(data), 50000, replace=False)])
        else:
            gmm.fit(data)

        if self.verbose > 1:
            print('Input statistics computed')



        initial_means = tf.Variable(gmm.means_.reshape(self.COMPONENTS, self.DIMENSIONS),
                                             dtype=tf.float32, name="means")

        initial_covariances = tf.Variable(gmm.covariances_.reshape(self.COMPONENTS,
                                                       self.DIMENSIONS,
                                                       self.DIMENSIONS),
                                                   dtype=tf.float32, name="covariances")

        initial_weights = tf.Variable(gmm.weights_.reshape(self.COMPONENTS),
                                               dtype=tf.float32, name="weights")

        # trainable variables: component means, covariances, and weights
        self.means = initial_means
        self.covariances = initial_covariances
        self.weights = initial_weights
        self.eta = tf.Variable((1 + self.T0) ** -self.alpha_eta, dtype=tf.float32, name="eta")


    def computeLL(self, exp_log_shifted_sum, log_shift):
        # log-likelihood: objective function being maximized up to a TOLERANCE delta
        log_likelihood = tf.reduce_sum(input_tensor=tf.math.log(exp_log_shifted_sum)) + tf.reduce_sum(input_tensor=log_shift)

        return log_likelihood / tf.cast(tf.shape(input=self.input)[0] * tf.shape(input=self.input)[1],
                                                            tf.float32)

    def computeSampleLL(self, exp_log_shifted_sum, log_shift):
        # log-likelihood: objective function being maximized up to a TOLERANCE delta
        return tf.math.log(exp_log_shifted_sum) + log_shift



    def E_step(self):

        # E-step: recomputing responsibilities with respect to the current parameter values
        differences = tf.subtract(tf.expand_dims(self.input, 0), tf.expand_dims(self.means, 1))
        try:
            diff_times_inv_cov = tf.matmul(differences, tf.linalg.inv(
                self.covariances + 1e-6 * tf.eye(self.DIMENSIONS, batch_shape=[self.COMPONENTS], dtype=tf.float32)))
        except:  # If covariances matrices are not invertible, add some noise to recover
            diff_times_inv_cov = tf.matmul(differences, tf.linalg.inv(
                self.covariances + tf.eye(self.DIMENSIONS, batch_shape=[self.COMPONENTS], dtype=tf.float32)))

        sum_sq_dist_times_inv_cov = tf.reduce_sum(input_tensor=diff_times_inv_cov * differences, axis=2)

        try:
            log_coefficients = tf.expand_dims(self.ln2piD + tf.math.log(
                tf.linalg.det(self.covariances),
                1))
        except:
            # If batch_size is two small regarding DIMENSIONS size,
            # covariances could be not low rank and so not invertible and so matrix_determinant is infinite
            log_coefficients = tf.expand_dims(self.ln2piD + tf.math.log(
                tf.linalg.det(self.covariances + 1e-6 * tf.eye(self.DIMENSIONS, batch_shape=[self.COMPONENTS],
                                                                       dtype=tf.float32))),
                                              1)
        log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)
        log_weighted = log_components + tf.expand_dims(tf.math.log(self.weights), 1)
        log_shift = tf.expand_dims(tf.reduce_max(input_tensor=log_weighted, axis=0), 0)
        exp_log_shifted = tf.exp(log_weighted - log_shift)
        exp_log_shifted_sum = tf.reduce_sum(input_tensor=exp_log_shifted, axis=0)
        gamma = exp_log_shifted / exp_log_shifted_sum
        gamma_sum = tf.reduce_sum(input_tensor=gamma, axis=1)
        gamma_weighted = gamma / tf.expand_dims(gamma_sum, 1)

        return log_shift, exp_log_shifted_sum, gamma_weighted, gamma_sum

    def M_step(self, log_shift, exp_log_shifted_sum, gamma_weighted, gamma_sum):

        # M-step: maximizing parameter values with respect to the computed responsibilities
        means_ = tf.reduce_sum(input_tensor=tf.expand_dims(self.input, 0) * tf.expand_dims(gamma_weighted, 2), axis=1)
        differences_ = tf.subtract(tf.expand_dims(self.input, 0), tf.expand_dims(means_, 1))
        sq_dist_matrix = tf.matmul(tf.expand_dims(differences_, 3), tf.expand_dims(differences_, 2))
        covariances_ = tf.reduce_sum(input_tensor=sq_dist_matrix * tf.expand_dims(tf.expand_dims(gamma_weighted, 2), 3), axis=1)
        weights_ = gamma_sum / tf.cast(tf.shape(input=self.input)[0], tf.float32)

        # applying prior to the computed covariances
        covariances_ *= tf.expand_dims(tf.expand_dims(gamma_sum, 1), 2)
        covariances_ += tf.expand_dims(tf.linalg.tensor_diag(tf.fill([self.DIMENSIONS], 2.0 * self.beta)), 0)
        covariances_ /= tf.expand_dims(tf.expand_dims(gamma_sum + (2.0 * (self.alpha + 1.0)), 1), 2)

        # log-likelihood: objective function being maximized up to a TOLERANCE delta
        self.mean_log_likelihood = self.computeLL(exp_log_shifted_sum, log_shift)

        # updating the parameters by new values
        train_step = tf.group(
            self.means.assign(self.means * (1 - self.eta) + self.eta * means_),
            self.covariances.assign(self.covariances * (1 - self.eta) + self.eta * covariances_),
            self.weights.assign(self.weights * (1 - self.eta) + self.eta * weights_)
        )

        return train_step



    def training_step(self):

        log_shift, exp_log_shifted_sum, gamma_weighted, gamma_sum = self.E_step()

        return self.M_step(log_shift, exp_log_shifted_sum, gamma_weighted, gamma_sum)

    def fit(self, data):

        # Init step
        self.initialisation(data=data)

        train_step = self.training_step()

        # initializing trainable variables
        batch_idx = np.random.choice(range(len(data)), size=self.BATCH_SIZE, replace=False)

        previous_likelihood = -np.inf

        # training loop
        for step in range(self.TRAINING_STEPS): #TRAINING_STEPS is only the maximal number of iterations
            if step < 10:
                self.eta.assign(self.eta / 2)
            else:
                self.eta.assign((step + self.T0) ** - self.alpha_eta)

            # executing a training step and
            # fetching evaluation information

            # Online (incremental EM)
            batch_idx = np.random.choice(range(len(data)), size=self.BATCH_SIZE, replace=False)
            batch = data[batch_idx]

            log_shift, exp_log_shifted_sum, gamma_weighted, gamma_sum = self.E_step()

            current_likelihood = self.computeLL(exp_log_shifted_sum, log_shift)

            if step > 0:
                # computing difference between consecutive likelihoods
                difference = current_likelihood - previous_likelihood
                if self.verbose > 0:
                    print("{0}:\tmean-likelihood {1:.8f}\tdifference {2}".format(
                        step, current_likelihood, difference))

                # stopping if TOLERANCE reached
                if difference <= self.TOLERANCE:
                    break
            else:
                if self.verbose > 0:
                    print("{0}:\tmean-likelihood {1:.8f}".format(
                        step, current_likelihood))

            previous_likelihood = current_likelihood

            if step == self.TRAINING_STEPS:
                try:
                    raise ConvergenceException(current_likelihood)
                except ConvergenceException as e:
                    print('EM has not converged. Mean likelihood value is :', str(e.value))


    def predict_proba(self, data):

        log_shift, exp_log_shifted_sum, gamma_weighted, gamma_sum = self.E_step()

        ll = self.computeLL(exp_log_shifted_sum, log_shift)

        return ll, gamma_weighted

    def loglikelihood_samples(self, data):
        log_shift, exp_log_shifted_sum, gamma_weighted, gamma_sum = self.sess.run(self.E_step(),
                                                                                  feed_dict={self.input: data})

        ll = self.sess.run(self.computeSampleLL(exp_log_shifted_sum, log_shift), feed_dict={self.input: data})

        return ll




