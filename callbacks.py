import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
import scipy.io as sio


class save_each_epoch(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        print('Model Saved at: ' + self.checkpoint_dir)
        self.model.save_weights(self.checkpoint_dir)


class PrintCA(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        super(PrintCA, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        ca_c = tf.squeeze(self.model.get_layer('Initialization_CASSI').get_weights())
        ca_m = tf.squeeze(self.model.get_layer('Initialization_MCFA').get_weights())
        sio.savemat(self.checkpoint_dir + '_' + str(epoch) + '.mat', {'cassi': ca_c.numpy(), 'mcfa': ca_m.numpy()})


class Aument_parameters(tf.keras.callbacks.Callback):
    def __init__(self, p_aum, p_step):
        super().__init__()
        self.p_aum = p_aum
        self.p_step = p_step

    def on_epoch_end(self, epoch, logs=None):
        if (tf.math.floormod(epoch, self.p_step) == (self.p_step - 1)):
            param = self.model.layers[1].my_regularizer.parameter
            param = tf.keras.backend.get_value(param)
            self.model.layers[1].my_regularizer.parameter.assign(param * self.p_aum)
            self.model.layers[2].my_regularizer.parameter.assign(param * self.p_aum)

        print('Binary regularizer parameter: ',
              tf.keras.backend.get_value(self.model.layers[1].my_regularizer.parameter))


class PlotRecovery(tf.keras.callbacks.Callback):
    def __init__(self, image_path, save_path, im_dict, n_stages, im_dims):
        self.image_path = image_path
        self.save_path = save_path
        self.im_dict = im_dict
        self.n_stages = n_stages
        self.im_dims = im_dims
        super(PlotRecovery, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        Im = sio.loadmat(self.image_path)[self.im_dict]
        print(Im.shape, self.im_dims)
        Im = tf.expand_dims(tf.image.resize(Im, [self.im_dims[0], self.im_dims[0], 31]), 0)
        Im = Im[:, :, :, 5:-6]
        X = self.model(Im)
        fig = plt.figure()
        fig.subplots_adjust(hspace=.05, wspace=.001)
        ax = fig.add_subplot(4, 3, 1)
        ax.imshow(np.squeeze(Im.numpy()[:, :, :, [15, 10, 5]]) / np.max(Im))
        ax.set_title('GT')
        ax.axis('off')

        for idx in range(10):
            ax = fig.add_subplot(4, 3, idx + 2)
            ax.imshow(np.squeeze(X[idx].numpy()[:, :, :, [15, 10, 5]]) / np.max(X[idx]))
            ax.set_title('Iteration  ' + str(idx))
            ax.axis('off')

        plt.savefig(self.save_path + '_' + str(epoch) + '.png')

