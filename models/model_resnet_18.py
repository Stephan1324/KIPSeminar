from models.model_manager import ModelManager
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, \
    GlobalAveragePooling2D, Dense, ReLU, Add
from keras.models import Model


class ModelResnet18(ModelManager):

    def __init__(self, batch_size: int = 32,
                 epochs: int = 10, initial_learningrate=0.01, hsv=False,
                 class_labels=['KGT_noDefect', 'KGT_pitting'],
                 img_height=150, img_width=150, img_channels=1):
        super().__init__(batch_size, epochs, initial_learningrate, hsv,
                         img_height, img_width, img_channels, class_labels)

        self.input_shape = (img_height, img_width, self.img_channels)
        self.model_type = 'resnet_18'
        self.model = self.create_resnet_18()

    def create_resnet_18(self):

        def conv_block(inputs, filters, kernel_size, strides=1):
            x = Conv2D(filters, kernel_size, strides=strides,
                       padding='same')(inputs)
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

        def ResNet18():
            inputs = Input(shape=self.input_shape)
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
        model = ResNet18(self.input_shape)

        return model
