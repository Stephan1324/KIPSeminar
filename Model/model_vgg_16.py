from Model.model_manager import ModelManager
from keras.applications import VGG16
from keras.layers import Dropout, Input, GlobalAveragePooling2D, Dense
from keras.regularizers import l2
from keras.models import Model


class ModelVgg16(ModelManager):

    def __init__(self, batch_size: int = 32,
                 epochs: int = 10, initial_learningrate=0.01, hsv=False,
                 img_height=150, img_width=150, img_channels=1,
                 trainable_layer_count=2):

        super().__init__(batch_size, epochs, initial_learningrate, hsv,
                         img_height, img_width, img_channels)

        self.input_shape = (img_height, img_width, img_channels)
        self.trainable_layer_count = trainable_layer_count
        self.model_type = 'vgg_16'
        self.model = self.create_vgg_16()

    def create_vgg_16(self):

        input_tensor = Input(self.input_shape)
        base_model = VGG16(include_top=False,
                           weights='imagenet', input_tensor=input_tensor)

        if self.trainable_layer_count == "all":
            for layer in base_model.layers:
                layer.trainable = True
        else:
            for layer in base_model.layers[:self.trainable_layer_count]:
                layer.trainable = False
            for layer in base_model.layers[self.trainable_layer_count:]:
                layer.trainable = True

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dropout(0.5)(x)
        final_output = Dense(1, activation='sigmoid', name='final_output')(x)
        model = Model(input_tensor, final_output)

        return model
