import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score

img_width, img_height = 256, 256
batch_size = 64

class FusionNetwork(tf.keras.Model):
    def __init__(self, num_classes):
        super(FusionNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        combined = tf.concat([inputs[0], inputs[1]], axis=1)  # 注意这里的变更，直接使用列表
        x = self.dense1(combined)
        return self.dense2(x)

def build_fusion_model(num_classes):
    model = FusionNetwork(num_classes=num_classes)
    return model

# 加载IMU和EMG模型
imu_model = tf.keras.models.load_model('E:\\master-2\\madgewick_filter\\training_model\\checkpoint\\imu_best_model.h5')
emg_model = tf.keras.models.load_model('E:\\master-2\\madgewick_filter\\training_model\\checkpoint\\emg_best_model.h5')

# 初始化fusion模型，以确保其具有正确的输入形状
num_classes = 7  # 根据您的实际情况调整
input_shape_imu = (7,)  # 假设从imu_model获取的特征维度
input_shape_emg = (7,)  # 假设从emg_model获取的特征维度
fusion_model = build_fusion_model(num_classes)
# 创建一个假的输入以初始化模型层次结构
fake_input_imu = tf.random.normal([batch_size] + list(input_shape_imu))
fake_input_emg = tf.random.normal([batch_size] + list(input_shape_emg))
_ = fusion_model([fake_input_imu, fake_input_emg])

# 加载模型权重
fusion_model.load_weights('E:\\master-2\\madgewick_filter\\training_model\\checkpoint\\fusion_model_weights_final.h5')

datagen = ImageDataGenerator(rescale=1./255)

# 测试数据生成器设置
test_imu_data_dir = 'E:/master-2/madgewick_filter/test_data_imu_pic/'
test_emg_data_dir = 'E:/master-2/madgewick_filter/SplitEMG_test_data_20240312/'
test_imu_labels_csv_path = 'E:/master-2/madgewick_filter/test_data_imu_pic/test_imu_labels.csv'
test_emg_labels_csv_path = 'E:/master-2/madgewick_filter/SplitEMG_test_data_20240312/test_emg_labels.csv'

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

# 执行测试
test_steps = min(len(test_imu_generator), len(test_emg_generator))
fused_preds = []
labels = []

for _ in range(test_steps):
    imu_batch, imu_labels = test_imu_generator.next()
    emg_batch, emg_labels = test_emg_generator.next()

    # 获取模型的输出，不再使用索引[1]
    imu_features = imu_model.predict(imu_batch)
    emg_features = emg_model.predict(emg_batch)

    # 确保特征为正确形状
    if imu_features.ndim == 1:
        imu_features = np.expand_dims(imu_features, axis=0)
    if emg_features.ndim == 1:
        emg_features = np.expand_dims(emg_features, axis=0)

    # 使用融合模型进行预测
    fused_pred = fusion_model.predict([imu_features, emg_features])

    fused_preds.extend(np.argmax(fused_pred, axis=1))
    labels.extend(np.argmax(imu_labels, axis=1))  # 假设IMU和EMG的标签是相同的

# 计算和打印性能指标
fused_preds = np.array(fused_preds)
labels = np.array(labels)
print('Classification Report:')
print(classification_report(labels, fused_preds))
print('Accuracy:', accuracy_score(labels, fused_preds))
