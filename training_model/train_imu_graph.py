from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
# from model import imu_graph_model  # Uncomment if using custom model
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')


labels_csv_path = 'E:/master-2/madgewick_filter/train_data/train_reach/output_imu_labels.csv'
df_labels = pd.read_csv(labels_csv_path)

data_dir = 'E:/master-2/madgewick_filter/train_data/train_reach/'

img_width, img_height = 256, 256
epochs = 100
batch_size = 1024

# Splitting data into 80% train and 20% validation
train_df, validate_df = train_test_split(df_labels, test_size=0.2, random_state=42, shuffle=True)

# Data augmentation and preparation
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_validate = ImageDataGenerator(rescale=1./255)

train_generator = datagen_train.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # Or 'binary' if you have 2 classes

validation_generator = datagen_validate.flow_from_dataframe(
    dataframe=validate_df,
    directory=data_dir,
    x_col='FilePath',
    y_col='Label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # Or 'binary' if you have 2 classes

# Define model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Change to match number of classes
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',  # Or 'binary_crossentropy'
              metrics=['accuracy'])

model_structure_file = 'model_structure.json'
model_json = model.to_json()
with open(model_structure_file, "w") as json_file:
    json_file.write(model_json)

# Model training
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator)

# Plotting training results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
