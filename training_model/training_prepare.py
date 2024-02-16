import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
from keras import regularizers
from tensorflow.keras.layers.experimental import preprocessing
from keras.models import Sequential
from keras.layers import Dropout
from model import create_model

filename = "imu1_train_data.csv"

df = pd.read_csv("../train_data_ori/" + filename)

index = range(1, len(df['IMU1_AccX']) + 1)

# plt.rcParams["figure.figsize"] = (20,10)

# plt.plot(index, df['IMU1_AccX'], 'g.', label='x', linestyle='solid', marker=',')
# plt.plot(index, df['IMU1_AccY'], 'b.', label='y', linestyle='solid', marker=',')
# plt.plot(index, df['IMU1_AccZ'], 'r.', label='z', linestyle='solid', marker=',')
# plt.title("Acceleration")
# plt.xlabel("Sample #")
# plt.ylabel("Acceleration (G)")
# plt.legend()
# plt.show()

# plt.plot(index, df['IMU1_GyrX'], 'g.', label='x', linestyle='solid', marker=',')
# plt.plot(index, df['IMU1_GyrY'], 'b.', label='y', linestyle='solid', marker=',')
# plt.plot(index, df['IMU1_GyrZ'], 'r.', label='z', linestyle='solid', marker=',')
# plt.title("Gyroscope")
# plt.xlabel("Sample #")
# plt.ylabel("Gyroscope (deg/sec)")
# plt.legend()
# plt.show()

print(f"TensorFlow version = {tf.__version__}\n")

# Set a fixed random seed value, for reproducibility, this will allow us to get
# the same random numbers each time the notebook is run
# "Reproducibility" means the ability to run the same thing twice and get 
#the same results.

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

directory = "../train_data/random/"
files = os.listdir(directory)

SAMPLES_PER_GESTURE = 1
GESTURES = ["random"]
NUM_GESTURES = len(GESTURES)

ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)

inputs = []
outputs = []

for gesture_index, gesture in enumerate(GESTURES):
    print(f"Processing gesture '{gesture}'.")
    
    output = ONE_HOT_ENCODED_GESTURES[gesture_index]
    
    gesture_files = [file for file in files if gesture in file]
    print(f"\tFound {len(gesture_files)} files for gesture '{gesture}'.")

    for file in gesture_files:
        df = pd.read_csv(directory + file)
        
        normalized_df = df.apply(lambda x: (x + 4) / 8 if 'Acc' in x.name else (x + 2000) / 4000)
        flattened_data = normalized_df.values.flatten()
        
        inputs.append(flattened_data)
        outputs.append(output)

inputs = np.array(inputs)
outputs = np.array(outputs)

print("Data set parsing and preparation complete.")

# Randomize the order of the inputs, so they can be evenly distributed for training, testing, and validation
# https://stackoverflow.com/a/37710486/2020087
num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

# Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
inputs = inputs[randomize]
outputs = outputs[randomize]

# Split the recordings (group of samples) into three sets: training, testing and validation
TRAIN_SPLIT = int(0.6 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

print("Data set randomization and splitting complete.")

# build the model and train it
model = create_model(input_shape=(120,), num_classes=5)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(inputs_train, outputs_train, epochs=100, batch_size=16, validation_data=(inputs_validate, outputs_validate))

# increase the size of the graphs. The default size is (6,4).
plt.rcParams["figure.figsize"] = (20,10)

# graph the loss, the model above is configure to use "mean squared error" as the loss function
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
