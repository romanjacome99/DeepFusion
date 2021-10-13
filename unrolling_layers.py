
import numpy as np
from deep_prior_networks import *
from regularizers import *
from tensorflow.keras.constraints import NonNeg

class GradientCASSI(Layer):
    def __init__(self, input_dim=(512, 512, 31), HO=(0.25, 0.5, 0.25), ds=0.5, name=False, shots=1, batch_size=1,**kwargs):
    
        self.input_dim = input_dim    
        self.ds = ds
        self.shots = shots
        self.batch_size = batch_size
        self.HO = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(HO, 0), 0), -1), -1)        
        self.Md = int(input_dim[0]*ds)
        super(GradientCASSI, self).__init__(**kwargs)

    def call(self, inputs):

        X = inputs[0]
        y = inputs[1]
        H = inputs[2]

        L = self.input_dim[2]
        Md = self.Md
        HO = self.HO
        X = tf.image.resize(X,[Md,Md])
        X = tf.expand_dims(X, -1)
        X = tf.reduce_sum(tf.nn.conv3d(X, HO, strides=[1, 1, 1, 1, 1], padding='SAME'), axis=-1)
        X = tf.multiply(H, tf.expand_dims(X, -1))
        X = tf.pad(X, [[0, 0], [0, 0], [0, L - 1], [0, 0], [0, 0]])
        yn = None
        for i in range(L):
            Tempo = tf.roll(X, shift=i, axis=2)
            if yn is not None:
                yn = tf.concat([yn, tf.expand_dims(Tempo[:, :, :, i], -1)], axis=4)
            else:
                yn = tf.expand_dims(Tempo[:, :, :, i], -1)
        yn = tf.reduce_sum(yn, -1)
        yn = yn / tf.math.reduce_max(yn)
        r = yn - y
        Xn = None
        for i in range(L):
            Tempo = tf.roll(r, shift=-i, axis=2)
            if Xn is not None:
                Xn = tf.concat([Xn, tf.expand_dims(Tempo[:, :, 0:Md], -1)], axis=4)
            else:
                Xn = tf.expand_dims(Tempo[:, :, 0:Md], -1)
        Xn = tf.transpose(Xn, [0, 1, 2, 4, 3])
        X2 = None
        for i in range(self.shots):
            Aux2 = tf.expand_dims(Xn[:, :, :, :, i], -1)
            if X2 is not None:
                X2 = tf.concat([X2, tf.expand_dims(tf.reduce_sum(
                    tf.nn.conv3d_transpose(Aux2, HO, [1, Md, Md, L, 1], strides=[1, 1, 1, 1, 1], padding='SAME'),
                    axis=-1), -1)], 4)
            else:
                X2 = tf.expand_dims(tf.reduce_sum(
                    tf.nn.conv3d_transpose(Aux2, HO, [1, Md, Md, L, 1], strides=[1, 1, 1, 1, 1], padding='SAME'),
                    axis=-1), -1)

        X2 = tf.multiply(H, X2)
        X2 = tf.reduce_sum(X2, 4)
        X = X2 / tf.math.reduce_max(X2)
        
        return X



class GradeientMCFA(Layer):

    def __init__(self, input_dim=(512, 512, 31), dl=0.5, shots=1, batch_size=1,snr=30,**kwargs):
        self.input_dim = input_dim
        self.shots = shots
        self.batch_size = batch_size
        self.Ld = int(self.input_dim[-1] * dl)
        super(GradeientMCFA, self).__init__(**kwargs)

    def call(self, inputs):

        X = inputs[0]
        y = inputs[1]
        H = inputs[2]
        L = self.input_dim[2]
        Ld = self.Ld
        M = self.input_dim[0]

        X = tf.expand_dims(tf.expand_dims(
            tf.reshape(tf.image.resize(tf.expand_dims(tf.reshape(X, [M * M, L]), -1), [M * M, Ld]),
                       [M, M, Ld]), 0), -1)
        yn = tf.expand_dims(tf.reduce_sum(tf.multiply(H, X), -2), -1)
        r = yn - y
        Xn = None
        for _ in range(Ld):
            if Xn is not None:
                Xn = tf.concat([Xn, r], 4)
            else:
                Xn = r
        Xn = tf.transpose(Xn, [0, 1, 2, 4, 3])

        Xn = tf.multiply(H, Xn)
        Xn = tf.reduce_sum(Xn, 4)
        X = Xn / tf.math.reduce_max(Xn)
      
        return X    
      

class AdaptiveParameter(Layer):

    def __init__(self, init = 0.01, name='Lambda'):
            super(AdaptiveParameter, self).__init__(name=name)
            w_init = tf.keras.initializers.Constant(value=init)
            self.w = tf.Variable(
                initial_value=w_init(shape=()),
                trainable=True, constraint=NonNeg()
            )

    def call(self, inputs):
        return tf.multiply(self.w, inputs)
