# Keras_Mnist_CNN_v1

import tensorflow as tf
import numpy as np
import pandas as pd

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# import tensorflow as tf
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return        


def CNN():
    model = Sequential()
    model.add(Conv2D(
        filters = 16,
        kernel_size = (5, 5),
        padding = 'same',
        input_shape = (28, 28, 1),
        activation = 'relu')
    )
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(
        filters = 36,
        kernel_size = (5, 5),
        padding = 'same',
        activation = 'relu')
    )
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def main():
    np.random.seed(10)
    # sess = tf.compat.v1.Session() 
    solve_cudnn_error()

    (x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()

    x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
    x_Test4D = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')

    x_Train4D_normalize = x_Train4D / 255
    x_Test4D_normalize = x_Test4D / 255

    y_Train_OneHot = np_utils.to_categorical(y_Train)
    y_Test_OneHot = np_utils.to_categorical(y_Test)

    model = CNN()
    print(model.summary())

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['acc']) # acc = accuracy

    train_history = model.fit(
        x = x_Train4D_normalize,
        y = y_Train_OneHot,
        validation_split = 0.2,
        epochs = 10,
        batch_size = 300,
        verbose = 2
    )

    scores = model.evaluate(x_Test4D_normalize, y_Test_OneHot)
    print('\nScore: ', scores[1])

    prediction = np.argmax(model.predict(x_Test4D_normalize), axis=-1)
    print('\nPrediction: ', prediction[:10])

    crosstab = pd.crosstab(
        y_Test, prediction, rownames = ['label'], colnames = ['predict']
    )
    df = pd.DataFrame(crosstab)
    print(); print(df)
    return


if __name__=='__main__':
    main()
