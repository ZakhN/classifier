from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Гиперпараметры
batch_size = 64
# 10 категорий для изображений  (CIFAR-10)
num_classes = 10
# количество эпох для обучения
epochs = 30


def load_data():
    """
    Эта функция загружает набор данных CIFAR-10 dataset и делает предварительную обработку
    """

    def preprocess_image(image, label):
        # преобразуем целочисленный диапазон [0, 255] в диапазон действительных чисел [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # загружаем набор данных CIFAR-10, разделяем его на обучающий и тестовый
    ds_train, info = tfds.load("cifar10", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("cifar10", split="test", as_supervised=True)
    # повторять набор данных, перемешивая, предварительно обрабатывая, разделяем по пакетам
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
