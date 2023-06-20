from models.model_manager import ModelManager
import tensorflow as tf


class BaselineAdvanced(ModelManager):

    def __init__(self, batch_size: int = 32,
                 epochs: int = 10, initial_learningrate=0.01, hsv=False,
                 img_height=150, img_width=150, img_channels=1,
                 class_labels=['KGT_noDefect', 'KGT_pitting'],
                 regularization_rate=0.001):
        super().__init__(batch_size, epochs, initial_learningrate, hsv,
                         img_height, img_width, img_channels, class_labels)

        self.input_shape = (img_height, img_width, img_channels)
        self.regularization_rate = regularization_rate
        self.model_type = 'baseline_advanced'
        self.model = self.create_baseline_advance()

    def create_baseline_advance(self):
        kernel_regularizer = tf.keras.regularizers.l2(self.regularization_rate)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(
                                       self.regularization_rate),
                                   input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
