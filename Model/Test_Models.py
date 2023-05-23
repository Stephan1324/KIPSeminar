import tensorflow as tf
from keras.applications import VGG16
from keras.applications.resnet import ResNet50
from keras.layers import (Dropout)
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Add, \
    GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.regularizers import l2


# Klasse zum erstellen von Test Modellen, welche einfach aufgerufen werden können in der Klasse Model
class Test_Models:

    def __init__(self, img_height=150, img_width=150, img_channels=1):
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels

        # Baseline Modell aus VL
        self.baseline = Test_Models.create_baseline(img_height=self.img_height, img_width=self.img_width,
                                                    img_channels=self.img_channels)

        # Resnet50
        self.model_resnet_50 = Test_Models.create_resnet_50(img_height=self.img_height, img_width=self.img_width,
                                                            img_channels=self.img_channels, trainable_layer_count='all')

        # VGG16
        self.model_vgg_16 = Test_Models.create_vgg16(img_height=self.img_height, img_width=self.img_width,
                                                     img_channels=self.img_channels, trainable_layer_count='all')

        # VGG16
        self.model_resnet_18 = Test_Models.create_resnet_18(img_height=self.img_height, img_width=self.img_width,
                                                            img_channels=self.img_channels)

        # Dictionary der Modelle
        self.model_dict = {"baseline": self.baseline,
                           "resnet_50": self.model_resnet_50,
                           "vgg_16": self.model_vgg_16,
                           "resnet_18": self.model_resnet_18
                           }

    # Funktion welche das entsprechende Modell über das Dictionary zurück gibt
    def get_model(self, name='baseline'):
        return self.model_dict[name]

    # Funktion zur Erstellung des Baseline Modells aus der VL
    @classmethod
    def create_baseline(cls, img_height, img_width, img_channels):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   input_shape=(img_height, img_width, img_channels)),
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

    @classmethod
    def create_resnet_50(cls, img_height, img_width, img_channels, trainable_layer_count):
        input_tensor = Input(shape=(img_height, img_width, img_channels))
        base_model = ResNet50(include_top=False,
                              weights=None,
                              input_tensor=input_tensor)

        if trainable_layer_count == "all":
            for layer in base_model.layers:
                layer.trainable = True
        else:
            for layer in base_model.layers[:-trainable_layer_count]:  # Fixed the indexing
                layer.trainable = False
            for layer in base_model.layers[-trainable_layer_count:]:
                layer.trainable = True

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dropout(0.5)(x)
        final_output = Dense(1, activation='sigmoid', name='final_output')(x)
        model = Model(input_tensor, final_output)

        return model

    @classmethod
    def create_vgg16(cls, img_height, img_width, img_channels, trainable_layer_count):
        input_tensor = Input(shape=(img_height, img_width, img_channels))
        base_model = VGG16(include_top=False, weights=None, input_tensor=input_tensor)

        if trainable_layer_count == "all":
            for layer in base_model.layers:
                layer.trainable = True
        else:
            for layer in base_model.layers[:-trainable_layer_count]:
                layer.trainable = False
            for layer in base_model.layers[-trainable_layer_count:]:
                layer.trainable = True

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dropout(0.5)(x)
        final_output = Dense(1, activation='sigmoid', name='final_output')(x)
        model = Model(input_tensor, final_output)

        return model

    @classmethod
    def create_resnet_18(cls, img_height, img_width, img_channels, ):

        def conv_block(inputs, filters, kernel_size, strides=1):
            x = Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        def identity_block(inputs, filters, kernel_size, strides=1):
            x = conv_block(inputs, filters, kernel_size, strides=strides)
            x = Conv2D(filters, kernel_size, padding='same')(x)
            x = BatchNormalization()(x)

            if strides != 1 or inputs.shape[-1] != filters:
                shortcut = Conv2D(filters, 1, strides=strides)(inputs)
                shortcut = BatchNormalization()(shortcut)
            else:
                shortcut = inputs

            x = Add()([shortcut, x])
            x = ReLU()(x)
            return x

        def ResNet18(input_shape):
            inputs = Input(shape=input_shape)
            x = conv_block(inputs, 64, 7, strides=2)
            x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

            x = identity_block(x, 64, 3)
            x = identity_block(x, 64, 3)

            x = identity_block(x, 128, 3, strides=2)
            x = identity_block(x, 128, 3)

            x = identity_block(x, 256, 3, strides=2)
            x = identity_block(x, 256, 3)

            x = identity_block(x, 512, 3, strides=2)
            x = identity_block(x, 512, 3)

            x = GlobalAveragePooling2D()(x)
            outputs = Dense(1, activation='sigmoid')(x)

            model = Model(inputs, outputs)
            return model

        # Example usage
        input_shape = (img_height, img_width, img_channels)
        model = ResNet18(input_shape)

        return model
