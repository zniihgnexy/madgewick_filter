import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

filename = "imu1_train_data.csv"

df = pd.read_csv("../train_data_ori/" + filename)

index = range(1, len(df['IMU1_AccX']) + 1)

plt.rcParams["figure.figsize"] = (20,10)

plt.plot(index, df['IMU1_AccX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['IMU1_AccY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['IMU1_AccZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Acceleration")
plt.xlabel("Sample #")
plt.ylabel("Acceleration (G)")
plt.legend()
plt.show()

plt.plot(index, df['IMU1_GyrX'], 'g.', label='x', linestyle='solid', marker=',')
plt.plot(index, df['IMU1_GyrY'], 'b.', label='y', linestyle='solid', marker=',')
plt.plot(index, df['IMU1_GyrZ'], 'r.', label='z', linestyle='solid', marker=',')
plt.title("Gyroscope")
plt.xlabel("Sample #")
plt.ylabel("Gyroscope (deg/sec)")
plt.legend()
plt.show()

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

print(f"Processing {len(files)} files.")

SAMPLES_PER_GESTURE = 200
GESTURES = ["random"]
NUM_GESTURES = len(GESTURES)

# create a one-hot encoded matrix that is used in the output
ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)

inputs = []
outputs = []

for gesture_index, gesture in enumerate(GESTURES):
    print(f"Processing gesture '{gesture}'.")
    
    output = ONE_HOT_ENCODED_GESTURES[gesture_index]
    
    gesture_files = [file for file in files if gesture in file]
    print("\t", len(gesture_files), "files found.")
    
    for file in gesture_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        
        num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)
        print(f"\tFile {file} has {num_recordings} recordings of the {gesture} gesture.")
        
        for i in range(num_recordings):
            tensor = []
            for j in range(SAMPLES_PER_GESTURE):
                index = i * SAMPLES_PER_GESTURE + j
                tensor += [
                    (df['IMU1_AccX'].iloc[index] + 4) / 8,
                    (df['IMU1_AccY'].iloc[index] + 4) / 8,
                    (df['IMU1_AccZ'].iloc[index] + 4) / 8,
                    (df['IMU1_GyrX'].iloc[index] + 2000) / 4000,
                    (df['IMU1_GyrY'].iloc[index] + 2000) / 4000,
                    (df['IMU1_GyrZ'].iloc[index] + 2000) / 4000
                ]

            inputs.append(tensor)
            outputs.append(output)

inputs = np.array(inputs)
outputs = np.array(outputs)

print("Data set parsing and preparation complete.")