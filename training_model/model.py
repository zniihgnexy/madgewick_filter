from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_shape, num_classes):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Use 'softmax' for classification tasks

    return model
