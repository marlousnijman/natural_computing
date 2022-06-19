import tensorflow as tf
from tensorflow.keras import layers, models


def DQN_Chrabaszcz_model(input_shape=(84, 84, 4), n_actions=4):
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

def DQN_Rodrigues_model(input_shape=(84, 84, 4), n_actions=4):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(8, (7, 7), strides=4, input_shape=input_shape))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Conv2D(16, (3, 3), strides=1))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))

    model.add(layers.Flatten())

    # Output layer
    model.add(layers.Dense(400)) 
    model.add(layers.Activation('relu'))

    model.add(layers.Dense(n_actions)) 
    model.add(layers.Activation('relu'))

    return model

def DQN_Mnih_model(input_shape=(84, 84, 4), n_actions=4):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (8, 8), strides=4, input_shape=input_shape))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(64, (4, 4), strides=2))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(64, (3, 3), strides=1))
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())

    # Output layer
    model.add(layers.Dense(512)) 
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(n_actions)) 

    return model