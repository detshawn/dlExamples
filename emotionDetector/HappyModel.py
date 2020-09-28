from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.backend import int_shape

def Block(X, n):
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(8*pow(2, n), (7-(2*n), 7-(2*n)), strides=(1, 1), name='conv'+str(n))(X)
    X = BatchNormalization(axis=3, name='bn'+str(n))(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool'+str(n))(X)

    return X

def HappyModel(input_shape):
    """
    :param
     input_shape: The height, weight and channels as a tuple.
    :return:
    """
    numBlocks = 3

    # input placeholder
    X_input = Input(input_shape)

    # zero-padding
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    for n in range(numBlocks):
        X = Block(X, n)
        # print("post-block{:d}: {}".format(n, X.shape))

    # FLATTEN X
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc0')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model
