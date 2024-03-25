import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
import curses

img_width, img_height = 256, 256
batch_size = 32

label_mapping = {
    0: ('baseline', 'perfect posture'),
    1: ('handlebar drop', 'over extended'),
    2: ('handlebar drop', 'under extended'),
    3: ('handlebar reach', 'over extended'),
    4: ('handlebar reach', 'under extended'),
    5: ('saddle height', 'over extended'),
    6: ('saddle height', 'under extended'),
}

# print_out_label = {
#     'handlebar drop': {
#         0: 'handlebar drop ok',
#         1: 'handlebar drop over extended',
#         2: 'handlebar drop under extended'
#     },
#     'handlebar reach': {
#         0: 'handlebar reach ok',
#         3: 'handlebar reach over extended',
#         4: 'handlebar reach under extended'
#     },
#     'saddle': {
#         0: 'saddle height ok',
#         5: 'saddle height over extended',
#         6: 'saddle height under extended'
#     }
# }

print_out_label = {
    'handlebar drop': ['baseline', 'over extended', 'under extended'],
    'handlebar reach': ['baseline', 'over extended', 'under extended'],
    'saddle height': ['baseline', 'over extended', 'under extended'],
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_table(label_names):
    clear_screen()  # 清除屏幕
    print("{:<20} | {:<15}".format("Category", "Status"))  # 打印表头
    print("-" * 37)
    for category, status in label_names:
        print("{:<20} | {:<15}".format(category, status))

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

imu_model = tf.keras.models.load_model('E:\\master-2\\madgewick_filter\\training_model\\checkpoint\\imu_best_model.h5')
emg_model = tf.keras.models.load_model('E:\\master-2\\madgewick_filter\\training_model\\checkpoint\\emg_best_model.h5')

num_classes = 7
input_shape_imu = (7,)
input_shape_emg = (7,)
fusion_model = build_fusion_model(num_classes)

fake_input_imu = tf.random.normal([batch_size] + list(input_shape_imu))
fake_input_emg = tf.random.normal([batch_size] + list(input_shape_emg))
_ = fusion_model([fake_input_imu, fake_input_emg])

fusion_model.load_weights('E:\\master-2\\madgewick_filter\\training_model\\checkpoint\\fusion_model_weights_final.h5')

datagen = ImageDataGenerator(rescale=1./255)

test_imu_data_dir = 'E:/master-2/madgewick_filter/test_data_imu_pic/'
test_emg_data_dir = 'E:/master-2/madgewick_filter/SplitEMG_test_data_20240312/'
test_imu_labels_csv_path = 'E:/master-2/madgewick_filter/test_data_imu_pic/test_imu_labels.csv'
test_emg_labels_csv_path = 'E:/master-2/madgewick_filter/SplitEMG_test_data_20240312/test_emg_labels.csv'

# test_imu_data_dir = sys.argv[1]
# test_emg_data_dir = sys.argv[2]
# test_imu_labels_csv_path = sys.argv[3]
# test_emg_labels_csv_path = sys.argv[4]

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

test_steps = min(len(test_imu_generator), len(test_emg_generator))
fused_preds = []
labels = []

for _ in range(test_steps):
    imu_batch, imu_labels = test_imu_generator.next()
    emg_batch, emg_labels = test_emg_generator.next()

    imu_features = imu_model.predict(imu_batch, verbose = 1)
    emg_features = emg_model.predict(emg_batch, verbose = 1)

    if imu_features.ndim == 1:
        imu_features = np.expand_dims(imu_features, axis=0)
    if emg_features.ndim == 1:
        emg_features = np.expand_dims(emg_features, axis=0)

    fused_pred = fusion_model.predict([imu_features, emg_features], verbose = 1)
    ##################### 只打印label的内容 #####################
    predicted_labels = np.argmax(fused_pred, axis=1)
    predicted_label_names = [label_mapping[label] for label in predicted_labels]
    time.sleep(3)
    print(predicted_labels)
    print(predicted_label_names)
    
    # get the most frequency number from predicted_labels
    print("predict the most")
    label_most = np.argmax(np.bincount(predicted_labels))
    print(label_most)

    label_most_name, label_most_status = label_mapping[label_most]

    status_updates = {
        'handlebar drop': 'baseline',
        'handlebar reach': 'baseline',
        'saddle height': 'baseline',
    }

    if label_most_name == 'handlebar drop':
        status_updates['handlebar drop'] = label_most_status
    elif label_most_name == 'handlebar reach':
        status_updates['handlebar reach'] = label_most_status
    elif label_most_name == 'saddle height':
        status_updates['saddle height'] = label_most_status
        
    

    print("-" * 62)
    print("{:<20} | {:<20} | {:<20}".format('Handlebar Drop', 'Handlebar Reach', 'Saddle Height'))

    formatted_status = [
        status_updates['handlebar drop'], 
        status_updates['handlebar reach'], 
        status_updates['saddle height']
    ]
    print("{:<20} | {:<20} | {:<20}".format(*formatted_status))
    print("-" * 62)



        
    ##################### 打印只代表头的内容 #####################
    # fused_preds.extend(np.argmax(fused_pred, axis=1))
    # labels.extend(np.argmax(imu_labels, axis=1))
    
    # predicted_labels = np.argmax(fused_pred, axis=1)
    # predicted_label_names = [(label_mapping[label][0], label_mapping[label][1]) for label in predicted_labels]

    # print("{:<20} | {:<15}".format("Category", "Status"))
    # print("-" * 37)
