from models.model_manager import ModelManager
import tensorflow as tf


class BaselineImproved(ModelManager):
  def __init__(self, batch_size=32, epochs=10, initial_learningrate=0.01, hsv=False,
                img_height=150, img_width=150, img_channels=1,
                class_labels=['KGT_noDefect', 'KGT_pitting']):
      super().__init__(batch_size, epochs, initial_learningrate, hsv,
                        img_height, img_width, img_channels, class_labels)

      self.input_shape = (img_height, img_width, self.img_channels)
      self.model_type = 'BaselineImproved'
      self.model = self.create_improved_model()

  def create_improved_model(self):
      l2_regularization = tf.keras.regularizers.l2(0.12)

      model = tf.keras.models.Sequential([

          tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D((2, 2)),

          tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2_regularization),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D((2, 2)),
          tf.keras.layers.MaxPooling2D(),

          tf.keras.layers.Conv2D(128, (3, 3), activation='tanh', kernel_regularizer=l2_regularization),
          tf.keras.layers.Dropout(0.3),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.MaxPooling2D((2, 2)),

          tf.keras.layers.Flatten(),

          tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
          tf.keras.layers.Dropout(0.5),

          tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification with sigmoid activation
      ])

      return model
