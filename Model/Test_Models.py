import tensorflow as tf


# Klasse zum erstellen von Test Modellen, welche einfach aufgerufen werden können in der Klasse Model
class Test_Models:

    def __init__(self, img_height=150, img_width=150, img_channels=1):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels

        # Baseline Modell aus VL
        self.baseline = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=(self.img_height, self.img_width, self.img_channels)),
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

        # Modell 1
        # Test aus Internet
        self.model_1 = tf.keras.models.Sequential([
            # Note the input shape is the desired size of the image 200x200 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   input_shape=(self.img_height, self.img_width, self.img_channels)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # # The fifth convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
            tf.keras.layers.Dense(1, activation='sigmoid')])

        # Modell 2
        # ---> Dummy
        self.model_2 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
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

        # Modell 3
        # ---> Dummy
        self.model_3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
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

        # Dictionary der Modelle
        self.model_dict = {"baseline": self.baseline,
                           "model_1": self.model_1,
                           "model_2": self.model_2,
                           "model_3": self.model_3,

                           }

    # Funktion welche das entsprechende Modell über das Dictionary zurück gibt
    def get_model(self, name='baseline'):
        return self.model_dict[name]
