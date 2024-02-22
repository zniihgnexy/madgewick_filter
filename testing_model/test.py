import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 步骤1：加载模型结构和权重
model_structure_file = '../training_model/model_structure.json'
with open(model_structure_file, 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

model_weights_file = '../training_model/checkpoint/model_weights.h5'
model.load_weights(model_weights_file)

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

test_data_dir = 'E:/master-2/madgewick_filter/test_data/imu/'
img_width, img_height = 256, 256
batch_size = 16

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # 或 'categorical'，取决于您的问题
    shuffle=False
)

print('Found %d images belonging to %d classes.' % (test_generator.samples, len(test_generator.class_indices)))

# 尝试获取第一批数据
x, y = next(test_generator)
print('Batch shape=%s, min=%.3f, max=%.3f' % (x.shape, x.min(), x.max()))

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

predictions = model.predict(test_generator)
