from Model.model_manager import ModelManager
import tensorflow as tf


class Baseline(ModelManager):

    def __init__(self, batch_size: int = 32,
                 epochs: int = 10, initial_learningrate=0.01, hsv=False,
                 img_height=150, img_width=150, img_channels=1):
        super().__init__(batch_size, epochs, initial_learningrate, hsv,
                         img_height, img_width, img_channels)

        self.input_shape = (img_height, img_width, img_channels)
        self.model_type = 'baseline'
        self.model = self.create_baseline()

    def create_baseline(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
