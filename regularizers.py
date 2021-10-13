import tensorflow as tf
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K


class Reg_Binary_0_1(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter=10):
        self.parameter = tf.keras.backend.variable(parameter, name='parameter')

    def __call__(self, x):
        regularization = self.parameter * (tf.reduce_sum(tf.multiply(tf.square(x), tf.square(1 - x))))
        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter))}


class Reg_Binary_Sigmoid(tf.keras.regularizers.Regularizer):
    def __init__(self, parameter_s=10, parameter_r=0.5):
        self.parameter_s = tf.keras.backend.variable(parameter_s, name='parameter_s')
        self.parameter_r = tf.keras.backend.variable(parameter_r, name='parameter_r')

    def __call__(self, x):
        regularization = self.parameter_r * tf.reduce_sum(
            tf.multiply(tf.square(tf.math.sigmoid(tf.multiply(x, self.parameter_s))),
                        tf.square(1 - tf.math.sigmoid(tf.multiply(x, self.parameter_s)))))
        return regularization

    def get_config(self):
        return {'parameter': float(tf.keras.backend.get_value(self.parameter_r))}


class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}
