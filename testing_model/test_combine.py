import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

img_width, img_height = 256, 256
batch_size = 64

imu_model = load_model('E:\master-2/madgewick_filter/training_model/checkpoint/imu_best_model.h5')
emg_model = load_model('E:\master-2/madgewick_filter/training_model/checkpoint/emg_best_model.h5')

datagen = ImageDataGenerator(rescale=1./255)

test_imu_data_dir = 'E:/master-2/madgewick_filter/test_data/test_reach/IMU/'
test_emg_data_dir = 'E:/master-2/madgewick_filter/test_data/test_reach/EMG/'

test_imu_labels_csv_path = 'E:/master-2/madgewick_filter/test_data/test_reach/IMU/output_imu_labels.csv'
test_emg_labels_csv_path = 'E:/master-2/madgewick_filter/test_data/test_reach/EMG/output_emg_labels.csv'

df_test_imu_labels = pd.read_csv(test_imu_labels_csv_path)
df_test_emg_labels = pd.read_csv(test_emg_labels_csv_path)

test_imu_generator = datagen.flow_from_dataframe(
    dataframe=df_test_imu_labels,
    directory=test_imu_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

test_emg_generator = datagen.flow_from_dataframe(
    dataframe=df_test_emg_labels,
    directory=test_emg_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

def fuse_and_predict(imu_probs, emg_probs):
    fused_probs = (imu_probs + emg_probs) / 2
    return np.argmax(fused_probs, axis=1)

test_steps = len(test_imu_generator)
imu_probs = []
emg_probs = []
labels = []

for i in range(test_steps):
    imu_batch, imu_labels = test_imu_generator.next()
    emg_batch, emg_labels = test_emg_generator.next()

    imu_pred = imu_model.predict(imu_batch)
    emg_pred = emg_model.predict(emg_batch)

    imu_probs.extend(imu_pred)
    emg_probs.extend(emg_pred)
    labels.extend(imu_labels)

imu_probs = np.array(imu_probs)
emg_probs = np.array(emg_probs)
labels = np.array(labels)
true_labels = np.argmax(labels, axis=1)

fused_preds = fuse_and_predict(imu_probs, emg_probs)
print('Classification Report:')
print(classification_report(true_labels, fused_preds))
print('Accuracy:', accuracy_score(true_labels, fused_preds))
