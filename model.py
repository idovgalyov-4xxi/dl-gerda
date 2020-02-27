# coding: utf-8
"""Core model module.

This includes a model class and some additional functions and parameter.
The model class does not include session management and parameters dumping.

The module is designed to work with TensorFlow version 1.15.
"""

from _imports import *
import os.path as op

from tensorflow.contrib.opt import LazyAdamOptimizer


# Default data types
DTYPE_FLOAT = tf.float64
DTYPE_INT = tf.int32

# Possible options are "/cpu:0" or "/gpu:0"
DEFAULT_DEVICE = "/cpu:0"
ALTERNATE_DEVICE = "/gpu:0"


def rmsle(y_pred, y_test):
    """Compute Root Mean Squared Logarithmic Error for predicted values.
    This function is provided for convenience.
    """
    assert len(y_test) == len(y_pred)
    return np.sqrt(
        np.mean(
            (np.log1p(y_pred) - np.log1p(y_test))**2
        )
    )


# Base frequencies expressed in hrs^-1
# (you can use this example to create other frequencies for your own task)
frequencies = np.array([
    # Secular
    0.0,
    # Annual
    0.00011408,
    # Semi-annual
    0.00022815,
    # Trimestral
    0.0003422318,
    # Quartal
    0.000456309,
    # Twice-monthly
    0.00068446366,
    # Monthly and semi-monthly
    0.001368927,
    0.001368927 * 2,
    # Weekly
    1 / (24 * 7),
    # Diurnal
    1 / 24,
    # Semidiurnal
    1 / 12,
    # 8-hours
    1 / 8,
    # 6-hours
    1 / 6,
    # 4-hours
    1 / 4,
    # 2-hours
    1 / 2,
])


class Model:
    """Main model class.
    The current model version uses simple MLP to compute amplitudes and phases
    for a set of chosen harmonics that approximates expected time series with
    a trigonometric series:

        y_pred(t) = sum[i=1..N]( C[i] * cos(w[i] * t + b[i] ) ),
    
    where C[i] and b[i] are predicted amplitudes and phases for a given
    epoch `t`.
    """
    TYPE = 'model12'

    def __init__(self, input_size,
                 time_feature_size,
                 frequencies,
                 N_hidden,
                 *,
                 use_residual_connection=True,
                 ):
        # Hidden layers size
        self.N_hidden = N_hidden

        # Total number of approximated harmonics
        self.frequencies = frequencies
        self.N_freq = len(frequencies)

        # Input size (except time epoch)
        self.input_size = input_size

        self.use_residual_connection = use_residual_connection

    def build(self):
        """Build model graph and define train operations."""

        # Inputs
        with tf.device(DEFAULT_DEVICE):
            # Raw input
            self.input = tf.placeholder(DTYPE_FLOAT, (None, self.input_size))
            print('Raw input shape:', self.inputs.shape)
            # Time epoch
            self.t = tf.placeholder(DTYPE_FLOAT, (None, 1))
            # Target values
            self.targets = tf.placeholder(DTYPE_FLOAT, (None, ))

        # First dense layer
        with tf.device(ALTERNATE_DEVICE):
            # For amplitudes
            self.c_layer_1 = tf.layers.Dense(
                units=self.N_hidden,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.tanh,
                name='layer_c_1',
            )
            # For phases
            self.b_layer_1 = tf.layers.Dense(
                units=self.N_hidden,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.tanh,
                name='layer_b_1',
            )

            self.c1 = self.c_layer_1(self.input)
            self.b1 = self.b_layer_1(self.input)
            print('First c/b_layer shape:', self.c1.shape)

        # Input for 2nd layer
        with tf.device(DEFAULT_DEVICE):
            if self.use_residual_connection:
                print("Residual connection enabled.")
                self.c_input_2 = tf.concat([self.input, self.c1], axis=1)
                self.b_input_2 = tf.concat([self.input, self.b1], axis=1)
            else:
                self.c_input_2 = self.c1
                self.b_input_2 = self.b1
            print('Input (for 2nd c/b_layer) shape:', self.c_input_2.shape)

        # Second dense layer
        with tf.device(ALTERNATE_DEVICE):
            # For amplitudes
            self.c_layer_2 = tf.layers.Dense(
                units=self.N_hidden,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.tanh,
                name='layer_c_2',
            )
            # For phases
            self.b_layer_2 = tf.layers.Dense(
                units=self.N_hidden,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.tanh,
                name='layer_b_2',
            )

            self.c2 = self.c_layer_2(self.c_input_2)
            self.b2 = self.b_layer_2(self.b_input_2)
            print('Second c/b_layer shape:', self.c2.shape)

        # Input for 3rd layer
        with tf.device(DEFAULT_DEVICE):
            if self.use_residual_connection:
                self.c_input_3 = tf.concat([self.input, self.c2], axis=1)
                self.b_input_3 = tf.concat([self.input, self.b2], axis=1)
            else:
                self.c_input_3 = self.c2
                self.b_input_3 = self.b2
            print('Input (for 3rd c/b_layer) shape:', self.c_input_3.shape)

        # Third dense layer
        with tf.device(ALTERNATE_DEVICE):
            # For amplitudes
            self.c_layer_3 = tf.layers.Dense(
                units=self.N_hidden,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.tanh,
                name='layer_c_3',
            )
            # For phases
            self.b_layer_3 = tf.layers.Dense(
                units=self.N_hidden,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.tanh,
                name='layer_b_3',
            )

            self.c3 = self.c_layer_3(self.c_input_3)
            self.b3 = self.b_layer_3(self.b_input_3)
            print('Third c/b_layer shape:', self.c3.shape)

        # Input for last layer
        with tf.device(DEFAULT_DEVICE):
            if self.use_residual_connection:
                self.c_input_last = tf.concat([self.input, self.c3], axis=1)
                self.b_input_last = tf.concat([self.input, self.b3], axis=1)
            else:
                self.c_input_last = self.c3
                self.b_input_last = self.b3
            print('Input (for last c/b_layer) shape:', self.c_input_last.shape)

        # Last layer (computing amplitudes, frequencies and phases)
        with tf.device(ALTERNATE_DEVICE):
            # For amplitudes
            self.c_layer_last = tf.layers.Dense(
                units=self.N_freq,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.math.softplus,
                name='layer_c_last',
            )
            # For phases
            self.b_layer_last = tf.layers.Dense(
                units=self.N_freq,
                kernel_initializer=tf.variance_scaling_initializer(),
                activation=tf.sigmoid,
                name='layer_b_last',
            )

            self.c = self.c_layer_last(self.c_input_last)
            self.w = tf.Variable(frequencies, trainable=False)
            self.b = self.b_layer_last(self.b_input_last)
        print('Last c/b_layer output shape:', self.c.shape)

        # Harmonic summator
        with tf.device(ALTERNATE_DEVICE):
            self.phase = (self.w * self.t + self.b) * (np.pi * 2)
            self.harmonic = self.c * tf.cos(self.phase)
            print('Time row shape:', self.t.shape)
            print('Harmonics shape:', self.harmonic.shape)

            self.output = tf.reduce_sum(self.harmonic, axis=1)

        # Computing output and metrics
        with tf.device(ALTERNATE_DEVICE):
            self.true_output = tf.expm1(tf.math.softplus(self.output))
            self.pseudo_targets = tf.log1p(self.targets)

            print('True output shape:', self.true_output.shape)
            print('True target shape:', self.targets.shape)

        with tf.device(DEFAULT_DEVICE):
            diff = self.output - self.pseudo_targets
            true_diff = self.true_output - self.targets
            print('Diff shape:', diff.shape)

        with tf.device(ALTERNATE_DEVICE):
            self.mae_metric = tf.reduce_mean(tf.abs(true_diff))
            self.mse_metric = tf.reduce_mean(tf.square(true_diff))
            self.rmse_metric = tf.sqrt(self.mse_metric)
            self.msle_metric = tf.reduce_mean(tf.square(
                tf.log1p(self.true_output) - tf.log1p(self.targets)
                ))
            self.rmsle_metric = tf.sqrt(self.msle_metric)
            print('MAE shape:', self.mae_metric.shape)
            print('MSE shape:', self.mse_metric.shape)
            print('RMSE shape:', self.rmse_metric.shape)
            print('RMSLE shape:', self.rmsle_metric.shape)


        # Loss function and optimizer
        with tf.device(ALTERNATE_DEVICE):
            # Chosen metric
            self.pseudo_mse_metric = tf.reduce_mean(
                tf.square(diff)
            )
            """
            self.pseudo_msle_metric = tf.reduce_mean(
                tf.square(
                    tf.math.softplus(self.output) - self.pseudo_targets
                )
            )
            """

            self.loss = self.pseudo_msle_metric

            self.learning_rate = tf.placeholder_with_default(1e-3, [])
            self.optimizer = LazyAdamOptimizer(
                learning_rate=self.learning_rate,
            )

        network_variables = [
            self.c_layer_1.kernel,
            self.c_layer_1.bias,
            self.b_layer_1.kernel,
            self.b_layer_1.bias,
            self.c_layer_2.kernel,
            self.c_layer_2.bias,
            self.b_layer_2.kernel,
            self.b_layer_2.bias,
            self.c_layer_3.kernel,
            self.c_layer_3.bias,
            self.b_layer_3.kernel,
            self.b_layer_3.bias,
            self.c_layer_last.kernel,
            self.c_layer_last.bias,
            self.b_layer_last.kernel,
            self.b_layer_last.bias,
        ]

        with tf.device(DEFAULT_DEVICE):
            self.train_op = self.optimizer.minimize(self.loss)
            self.train_op_network_only = self.optimizer.minimize(self.loss, var_list=network_variables)

        # Operation for resetting Adam parameters
        self.reset_optimizer_op = tf.variables_initializer(
            [self.optimizer.get_slot(var, name)
             for name in self.optimizer.get_slot_names()
             for var in tf.trainable_variables()]
            + list(self.optimizer._get_beta_accumulators())
        )

    def fit(self, feed_dict, session=None, mode='default'):
        if session is None:
            session = tf.get_default_session()

        all_for_train = {
            'c': self.c,
            'w': self.w,
            'b': self.b,
            'true_output': self.true_output,
            'mae': self.mae_metric,
            'rmse': self.rmse_metric,
            'rmsle': self.rmsle_metric,
            'loss': self.loss,
            'train_op': self.train_op,
        }
        if mode == 'network':
            all_for_train['train_op'] = self.train_op_network_only
        else:
            all_for_train['train_op'] = self.train_op

        return session.run(all_for_train, feed_dict=feed_dict)

    def predict(self, feed_dict, session=None):
        if session is None:
            session = tf.get_default_session()

        all_for_test = {
            'c': self.c,
            'w': self.w,
            'b': self.b,
            'output': self.output,
            'true_output': self.true_output,
            'mae': self.mae_metric,
            'rmse': self.rmse_metric,
            'rmsle': self.rmsle_metric,
            'loss': self.loss,
        }

        return session.run(all_for_test, feed_dict=feed_dict)

    def reset_optimizer(self, session=None):
        if session is None:
            session = tf.get_default_session()

        return session.run(self.reset_optimizer_op)

