import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def loss_unrolling(rho_s,rho_l):
    def lossimage_2(y_true, y_pred):
        mae = tf.keras.losses.MeanAbsoluteError()
        
        spatial_loss = tf.reduce_mean(1 - tf.image.ssim(y_pred, y_true, 1)) + tf.reduce_mean(1/np.prod(y_pred.shape)*tf.norm(y_true-y_pred,ord=1))
        a_b = tf.math.reduce_sum(tf.multiply(y_pred, y_true), axis=-1)
        mag_a = tf.sqrt(tf.reduce_sum(y_pred ** 2, axis=-1))
        mag_b = tf.sqrt(tf.reduce_sum(y_true ** 2, axis=-1))
        spectral_loss = tf.reduce_mean(tf.abs(a_b - tf.multiply(mag_a, mag_b)))
        val = rho_l * spectral_loss + rho_s * spatial_loss        return val

    return lossimage_2


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1))


def ergas_metric(y_true, y_pred):
    return tf.reduce_mean(rmse(y_true, y_pred) / tf.reduce_mean(y_true, axis=(1, 2, 3)))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_pred, y_true, 1))


def rmse(y_true, y_pred):
    true_norm = K.sqrt(K.sum(K.square(y_true), axis=-1))
    return K.mean(K.sqrt(tf.keras.losses.mean_squared_error(y_true, y_pred)) / true_norm)

