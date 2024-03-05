import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.losses import Loss

set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

train_losses = []
train_accuracies = []
fusion_accuracy = []

def fuse_and_predict(imu_probs, emg_probs):
    fused_probs = (imu_probs + emg_probs) / 2
    return np.argmax(fused_probs, axis=1)

def train_one_epoch(imu_model, emg_model, train_imu_gen, train_emg_gen, 
                    optimizer_imu, optimizer_emg, loss_fn, train_loss_metric_imu, train_accuracy_metric_imu, 
                    train_loss_metric_emg, train_accuracy_metric_emg, fusion_accuracy_metric, steps):

    train_loss_metric_imu.reset_states()
    train_accuracy_metric_imu.reset_states()
    train_loss_metric_emg.reset_states()
    train_accuracy_metric_emg.reset_states()
    fusion_accuracy_metric.reset_states()
    
    for step in range(steps):
        imu_batch, imu_labels = train_imu_gen.next()
        emg_batch, emg_labels = train_emg_gen.next()

        with tf.GradientTape() as tape:
            imu_predictions,_ = imu_model(imu_batch, training=True)
            loss_imu = loss_fn(imu_labels, imu_predictions)
        gradients_imu = tape.gradient(loss_imu, imu_model.trainable_variables)
        optimizer_imu.apply_gradients(zip(gradients_imu, imu_model.trainable_variables))
        train_loss_metric_imu.update_state(loss_imu)
        train_accuracy_metric_imu.update_state(imu_labels, imu_predictions)

        with tf.GradientTape() as tape:
            emg_predictions,_ = emg_model(emg_batch, training=True)
            loss_emg = loss_fn(emg_labels, emg_predictions)
        gradients_emg = tape.gradient(loss_emg, emg_model.trainable_variables)
        optimizer_emg.apply_gradients(zip(gradients_emg, emg_model.trainable_variables))
        train_loss_metric_emg.update_state(loss_emg)
        train_accuracy_metric_emg.update_state(emg_labels, emg_predictions)

        fused_predictions = (imu_predictions + emg_predictions) / 2
        fusion_accuracy_metric.update_state(imu_labels, fused_predictions)
        # fusion_accuracy.append(fusion_accuracy_metric.result().numpy())

        if step % 10 == 0:
            print(f'Step {step}, IMU Loss: {train_loss_metric_imu.result().numpy()}, IMU Accuracy: {train_accuracy_metric_imu.result().numpy()}')
            print(f'Step {step}, EMG Loss: {train_loss_metric_emg.result().numpy()}, EMG Accuracy: {train_accuracy_metric_emg.result().numpy()}')
            print(f'Step {step}, Fusion Accuracy: {fusion_accuracy_metric.result().numpy()}')
            fusion_accuracy.append(fusion_accuracy_metric.result().numpy())

    print(f'End of epoch, IMU Loss: {train_loss_metric_imu.result().numpy()}, IMU Accuracy: {train_accuracy_metric_imu.result().numpy()}')
    print(f'End of epoch, EMG Loss: {train_loss_metric_emg.result().numpy()}, EMG Accuracy: {train_accuracy_metric_emg.result().numpy()}')
    print(f'End of epoch, Fusion Accuracy: {fusion_accuracy_metric.result().numpy()}')

img_width, img_height = 256, 256
epochs_imu = 10
epochs_emg = 10
batch_size_imu = 64
batch_size_emg = 64

datagen = ImageDataGenerator(rescale=1./255)

imu_labels_csv_path = 'E:/master-2/madgewick_filter/train_data/train_reach/IMU/output_imu_labels.csv'
df_imu_labels = pd.read_csv(imu_labels_csv_path)
imu_data_dir = 'E:/master-2/madgewick_filter/train_data/train_reach/IMU/'
train_df_imu, validate_df_imu = train_test_split(df_imu_labels, test_size=0.2, random_state=42, shuffle=True)

train_imu_generator = datagen.flow_from_dataframe(
    dataframe=train_df_imu,
    directory=imu_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size_imu,
    class_mode='categorical')

validation_imu_generator = datagen.flow_from_dataframe(
    dataframe=validate_df_imu,
    directory=imu_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size_imu,
    class_mode='categorical')

emg_labels_csv_path = 'E:/master-2/madgewick_filter/train_data/train_reach/EMG/output_emg_labels.csv'
df_emg_labels = pd.read_csv(emg_labels_csv_path)
emg_data_dir = 'E:/master-2/madgewick_filter/train_data/train_reach/EMG/'
train_df_emg, validate_df_emg = train_test_split(df_emg_labels, test_size=0.2, random_state=42, shuffle=True)

train_emg_generator = datagen.flow_from_dataframe(
    dataframe=train_df_emg,
    directory=emg_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size_emg,
    class_mode='categorical')

validation_emg_generator = datagen.flow_from_dataframe(
    dataframe=validate_df_emg,
    directory=emg_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size_emg,
    class_mode='categorical')

# IMU
imu_input = Input(shape=(img_width, img_height, 3), name='imu_input')
x = Conv2D(16, (3, 3), activation='relu')(imu_input)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)
imu_dense = Dense(128, activation='relu')(x)
imu_output = Dense(3, activation='softmax', name='imu_output')(imu_dense)
imu_model = Model(inputs=imu_input, outputs=[imu_output, imu_dense])

# EMG
emg_input = Input(shape=(img_width, img_height, 3), name='emg_input')
y = Conv2D(16, (3, 3), activation='relu')(emg_input)
y = MaxPooling2D(2, 2)(y)
y = Conv2D(32, (3, 3), activation='relu')(y)
y = MaxPooling2D(2, 2)(y)
y = Conv2D(64, (3, 3), activation='relu')(y)
y = MaxPooling2D(2, 2)(y)
y = Flatten()(y)
emg_dense  = Dense(128, activation='relu')(y)
emg_output = Dense(3, activation='softmax', name='emg_output')(emg_dense)
emg_model = Model(inputs=emg_input, outputs=[emg_output, emg_dense])

imu_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
emg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_imu = ModelCheckpoint(filepath='imu_best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint_emg = ModelCheckpoint(filepath='emg_best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

steps_per_epoch = min(len(train_imu_generator), len(train_emg_generator))

imu_batch, _ = next(iter(train_imu_generator))
emg_batch, _ = next(iter(train_emg_generator))

imu_predictions, _ = imu_model.predict(imu_batch)
emg_predictions, _ = emg_model.predict(emg_batch)

print("IMU predictions shape:", imu_predictions.shape)
print("EMG predictions shape:", emg_predictions.shape)


epochs = 100

optimizer_imu = tf.keras.optimizers.Adam()
optimizer_emg = tf.keras.optimizers.Adam()

train_loss_metric_imu = tf.keras.metrics.Mean()
train_accuracy_metric_imu = tf.keras.metrics.CategoricalAccuracy()
train_loss_metric_emg = tf.keras.metrics.Mean()
train_accuracy_metric_emg = tf.keras.metrics.CategoricalAccuracy()
fusion_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

# Assuming 'steps_per_epoch' is already defined correctly
steps_per_epoch = min(len(train_imu_generator), len(train_emg_generator))

# Now, call the function with all arguments
for epoch in range(epochs):
    print(f'Start of Epoch {epoch+1}')
    train_one_epoch(imu_model, emg_model, train_imu_generator, train_emg_generator, optimizer_imu, optimizer_emg, loss_fn, train_loss_metric_imu, train_accuracy_metric_imu, train_loss_metric_emg, train_accuracy_metric_emg, fusion_accuracy_metric, steps_per_epoch)

print("Fusion Accuracy:", fusion_accuracy)

import matplotlib.pyplot as plt
# After all episodes
plt.figure(figsize=[10,5])
plt.plot(np.arange(len(fusion_accuracy)), fusion_accuracy)
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.title('Fusion Accuracy over all episodes')
plt.show()

