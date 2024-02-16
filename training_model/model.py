from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

def create_imu_model (input_shape, num_classes):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Use 'softmax' for classification tasks

    return model

def build_imu_model(X_train, y_train, batch_size=64, epochs=100):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=108, kernel_size=5, padding='same',
                                activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape=(120,)),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same',
                                activation=tf.keras.layers.ReLU()),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same',
                                activation=tf.keras.layers.ReLU()),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=108, activation=tf.keras.layers.ReLU()),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=4, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['sparse_categorical_accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_train, y_train))
    model.summary()
    # 获得训练集和测试集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 绘制acc曲线
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # 绘制loss曲线
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()
