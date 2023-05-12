import tensorflow as tf


# Klasse zum erstellen von Test Modellen, welche einfach aufgerufen werden können in der Klasse Model
class Test_Models:

    def __init__(self, img_height=150, img_width=150, img_channels=1):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels

        # Baseline Modell
        self.baseline = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, img_channels)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        # Modell 1
        # ---> Dummy
        self.model_1 = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, img_channels)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        # Modell 2
        # ---> Dummy
        self.model_2 = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, img_channels)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        # Modell 3
        # ---> Dummy
        self.model_3 = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, img_channels)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

        # Dictionary der Modelle
        self.model_dict = {"baseline": self.baseline,
                           "model_1": self.model_1,
                           "model_2": self.model_2,
                           "model_3": self.model_3,

                           }

    # Funktion welche das entsprechende Modell über das Dictionary zurück gibt
    def get_model(self, name='baseline'):
        return self.model_dict[name]
