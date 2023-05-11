import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from CONFIG_GLOBAL import CONFIG_GLOBAL
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential


class Model:

    def __init__(self, model_type=None):
        self.batch_size = 32
        self.img_height = 150
        self.img_width = 150
        self.data_directory = CONFIG_GLOBAL.PATH_CLEANED_DATA_FOLDER

    def split(self, test_size=0.2):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_directory,
            validation_split=test_size,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_directory,
            validation_split=test_size,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size)

        class_names = self.train_ds.class_names
        self.num_classes = len(class_names)
        print(class_names)

    def model_building(self):
        # ---------- Model building ----------
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        for image_batch, labels_batch in self.train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        self.model = (Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes)
        ]))

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.build(input_shape=(self.batch_size, self.img_height, self.img_width, 3))
        self.model.summary()

    def fit(self):
        # ------ training ----------
        self.epochs = 10
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

    def evaluate(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    # def predict(self, input_value):
    #     if input_value == None:
    #         result = self.user_defined_model.fit(self.X_test)
    #     else:
    #         result = self.user_defined_model.fit(np.array([input_value]))
    #     return result
