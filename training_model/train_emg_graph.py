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

# Load EMG data
emg_labels_csv_path = 'E:/master-2/madgewick_filter/train_data/train_reach/EMG/output_emg_labels.csv'
df_emg_labels = pd.read_csv(emg_labels_csv_path)
emg_data_dir = 'E:/master-2/madgewick_filter/train_data/train_reach/EMG/'

img_width, img_height = 256, 256
epochs = 100
batch_size = 128

# Splitting EMG data
train_emg_df, validate_emg_df = train_test_split(df_emg_labels, test_size=0.2, random_state=42, shuffle=True)

# EMG data augmentation
datagen_emg_train = ImageDataGenerator(rescale=1./255)
datagen_emg_validate = ImageDataGenerator(rescale=1./255)

train_emg_generator = datagen_emg_train.flow_from_dataframe(
    dataframe=train_emg_df,
    directory=emg_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_emg_generator = datagen_emg_validate.flow_from_dataframe(
    dataframe=validate_emg_df,
    directory=emg_data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# EMG model architecture
emg_input = Input(shape=(img_width, img_height, 3))
y = Conv2D(16, (3, 3), activation='relu')(emg_input)
y = MaxPooling2D(2, 2)(y)
y = Dropout(0.2)(y)

y = Conv2D(16, (3, 3), activation='relu')(y)
y = MaxPooling2D(2, 2)(y)
y = Dropout(0.2)(y)

y = Conv2D(16, (3, 3), activation='relu')(y)
y = MaxPooling2D(2, 2)(y)
y = Flatten()(y)
y = Dense(64, activation='relu')(y)
y = Dropout(0.5)(y)
output = Dense(3, activation='softmax')(y)
emg_model = Model(inputs=emg_input, outputs=output)

# Compile EMG model
emg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train EMG model
history = emg_model.fit(
    train_emg_generator,
    steps_per_epoch=len(train_emg_generator),
    epochs=epochs,
    validation_data=validation_emg_generator,
    validation_steps=len(validation_emg_generator)
)

# Plotting training results for EMG
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('EMG Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('EMG Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
