from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from model import imu_graph_model

labels_csv_path = 'E:/master-2/madgewick_filter/train_data/normalized_images/labels.csv'
df_labels = pd.read_csv(labels_csv_path)

data_dir = 'E:/master-2/madgewick_filter/train_data/normalized_images/'

img_width, img_height = 256, 256
epochs = 100
batch_size = 256

train_df, validate_df = train_test_split(df_labels, test_size=0.3, random_state=42, shuffle=True)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=data_dir,
    x_col='filename',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=data_dir,
    x_col='filename',
    y_col='label',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # 或 'binary' 如果有2个类别
    subset='validation')

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
    Dense(1, activation='sigmoid')
])

# model = imu_graph_model(1, img_width, img_height)

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])

# checkpoint = ModelCheckpoint('./checkpoint/model_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
model_structure_file = 'model_structure.json'

model_json = model.to_json()
with open(model_structure_file, "w") as json_file:
    json_file.write(model_json)

# 模型定义、编译和训练的部分保持不变

history = model.fit(train_generator,
            epochs=epochs,
            validation_data=validation_generator)
            # callbacks=[checkpoint])
print(history.history.keys())

# 注意：模型权重已保存在'model_weights.h5'，模型结构保存在'model_structure.json'
# 绘制训练和验证的准确率变化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 绘制训练和验证的损失值变化
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss over epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
plt.show()