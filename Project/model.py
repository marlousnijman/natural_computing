import tensorflow as tf
from tensorflow.keras import layers, models


def DQN(input_shape=(84, 84, 4), n_actions=4):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (8, 8), strides=4, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))

    model.add(layers.Conv2D(64, (4, 4), 2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))

    model.add(layers.Conv2D(64, (3, 3), 1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))

    model.add(layers.Flatten())

    # Dense layer
    model.add(layers.Dense(512)) 
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))

    # Output layer
    model.add(layers.Dense(n_actions)) 
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))

    return model