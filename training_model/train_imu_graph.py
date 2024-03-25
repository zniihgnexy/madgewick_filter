from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Load IMU data
# imu_labels_csv_path = 'E:/master-2/madgewick_filter/train_data/train_reach/IMU/output_imu_labels.csv'
# df_imu_labels = pd.read_csv(imu_labels_csv_path)
# imu_data_dir = 'E:/master-2/madgewick_filter/train_data/train_reach/IMU/'

imu_labels_csv_path = 'E:/master-2/madgewick_filter/train_data_imu_pic/train_imu_labels.csv'
df_imu_labels = pd.read_csv(imu_labels_csv_path)
imu_data_dir = 'E:/master-2/madgewick_filter/train_data_imu_pic/'

img_width, img_height = 256, 256
epochs = 50
batch_size = 64

# Splitting IMU data
train_df, validate_df = train_test_split(df_imu_labels, test_size=0.2, random_state=42, shuffle=True)

# IMU data augmentation
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_validate = ImageDataGenerator(rescale=1./255)

train_imu_generator = datagen_train.flow_from_dataframe(
    dataframe=train_df,
    directory=imu_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_imu_generator = datagen_validate.flow_from_dataframe(
    dataframe=validate_df,
    directory=imu_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# IMU model architecture
imu_input = Input(shape=(img_width, img_height, 3))
x = Conv2D(16, (3, 3), activation='relu')(imu_input)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = MaxPooling2D(2, 2)(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(7, activation='softmax')(x)
imu_model = Model(inputs=imu_input, outputs=output)

# Compile IMU model
imu_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_path = 'E:/master-2/madgewick_filter/training_model/checkpoint/imu_best_model.h5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

# Train IMU model
history = imu_model.fit(
    train_imu_generator,
    steps_per_epoch=len(train_imu_generator),
    epochs=epochs,
    validation_data=validation_imu_generator,
    validation_steps=len(validation_imu_generator),
    callbacks=[checkpoint]
)

# Plotting training results for IMU
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('IMU Accuracy over epochs', fontsize=18)  # 调整标题字体大小
plt.xlabel('Epoch', fontsize=15)  # 调整X轴标签字体大小
plt.ylabel('Accuracy', fontsize=15)  # 调整Y轴标签字体大小
plt.legend(fontsize=15)  # 调整图例字体大小

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('IMU Loss over epochs', fontsize=18)  # 调整标题字体大小
plt.xlabel('Epoch', fontsize=15)  # 调整X轴标签字体大小
plt.ylabel('Loss', fontsize=15)  # 调整Y轴标签字体大小
plt.legend(fontsize=15)  # 调整图例字体大小

plt.tight_layout()
plt.show()

