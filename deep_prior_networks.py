from tensorflow.keras.layers import *


def deep_prior_unet(X, Bands=20, Kernels_Size=(3, 3), num_filters=20, trainable=True, latent_space=None):
    conv_r1 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation=None,
                     trainable=trainable)(X)

    down1 = MaxPool2D(pool_size=(2, 2))(conv_r1)

    conv_r2 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation=None,
                     trainable=trainable)(down1)

    down2 = MaxPool2D(pool_size=(2, 2))(conv_r2)

    conv_r3 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu",
                     trainable=trainable)(down2)

    down3 = MaxPool2D(pool_size=(2, 2))(conv_r3)

    latent_space_1 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal',
                            activation="relu", trainable=trainable)(down3)  # latent space
    if latent_space is not None:
        latent_space_1 = Concatenate()([latent_space,latent_space_1])

    up1 = UpSampling2D(size=(2, 2))(latent_space_1)

    merge1 = Concatenate(axis=3)([up1, conv_r3])

    conv_r5 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu",
                     trainable=trainable)(merge1)

    up2 = UpSampling2D(size=(2, 2))(conv_r5)

    merge2 = Concatenate(axis=3)([up2, conv_r2])

    conv_r6 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu"
                     , trainable=trainable)(merge2)

    up3 = UpSampling2D((2, 2))(conv_r6)

    merge3 = Concatenate(axis=3)([up3, conv_r1])

    conv_r7 = Conv2D(num_filters, Kernels_Size, padding="same", kernel_initializer='he_normal', activation="relu"
                     , trainable=trainable)(merge3)

    res_op = Add()([X, conv_r7])

    conv_r8 = Conv2D(Bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None,
                     trainable=trainable)(res_op)

    return latent_space_1, conv_r8


def prior_highres(X, Bands=20, Kernels_Size=(3, 3), num_filters=20, trainable=True):
    Up1 = UpSampling2D([2, 2])(X)

    conv_r1 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(Up1)

    conv_r2 = Conv2D(2 * Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r1)

    Down1 = MaxPool2D(pool_size=[2, 2])(conv_r2)

    conv_r3 = Conv2D(2 * Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(Down1)

    conv_r4 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation="relu")(conv_r3)

    conv_r4 = BatchNormalization()(conv_r4)

    conv_r4 = Add()([X, conv_r4])

    conv_r5 = Conv2D(Bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)

    return conv_r5

def sota_prior(X, Bands=20, Kernels_Size=(3, 3), num_filters=20, trainable=True):

    conv_r1 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(X)

    conv_r2 = Conv2D(Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation='relu')(conv_r1)

    conv_r3 = Conv2D( Bands, (3, 3), padding="same", kernel_initializer='he_normal', activation=None)(conv_r2)

    conv_r4 = Add()([X, conv_r3])

    conv_r5 = Conv2D(Bands, (1, 1), padding="same", kernel_initializer='he_normal', activation=None)(conv_r4)

    return conv_r5


def residual_block(X, Bands=20, Kernels_Size=(3, 3), num_filters=20):
    X0 = Conv2D(num_filters, Kernels_Size, padding='same', kernel_initializer='he_normal', activation='relu')(X)
    X1 = BatchNormalization()(X0)
    X2 = Conv2D(num_filters, Kernels_Size, padding='same', kernel_initializer='he_normal', activation='relu')(X1)
    X3 = Add()([X0, X2])
    X4 = BatchNormalization()(X3)
    return X4


def initialization_network(X, Bands=20, Kernels_Size=(3, 3), num_filters=20):
    X0 = Conv2D(num_filters, Kernels_Size, padding='same', kernel_initializer='he_normal', activation='relu')(X)
    X1 = residual_block(X0, Bands=Bands, Kernels_Size=Kernels_Size, num_filters=num_filters)
    X2 = residual_block(X1, Bands=Bands, Kernels_Size=Kernels_Size, num_filters=num_filters)
    X3 = residual_block(X2, Bands=Bands, Kernels_Size=Kernels_Size, num_filters=num_filters)
    return X3
