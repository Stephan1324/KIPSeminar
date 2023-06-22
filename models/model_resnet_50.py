from models.model_manager import ModelManager
from keras.applications.resnet import ResNet50
from keras.layers import (Dropout, Input, GlobalAveragePooling2D, Dense)
from keras.models import Model
from keras.regularizers import l2


class ModelResnet50(ModelManager):

    def __init__(self, batch_size: int = 32,
                 epochs: int = 10, initial_learningrate=0.01, hsv=False,
                 img_height=150, img_width=150, img_channels=1,
                 class_labels=['KGT_noDefect', 'KGT_pitting'],
                 trainable_layer_count=12):

        super().__init__(batch_size, epochs, initial_learningrate, hsv,
                         img_height, img_width, img_channels, class_labels)

        self.input_shape = (img_height, img_width, self.img_channels)
        self.trainable_layer_count = trainable_layer_count
        self.model_type = 'resnet_50'
        self.model = self.create_resnet_50()

    def create_resnet_50(self):

        input_tensor = Input(self.input_shape)
        base_model = ResNet50(include_top=False,
                              weights="imagenet",
                              input_tensor=input_tensor)

        if self.trainable_layer_count == "all":
            for layer in base_model.layers:
                layer.trainable = True
        else:
            # Fixed the indexing
            for layer in base_model.layers[:-self.trainable_layer_count]:
                layer.trainable = False
            for layer in base_model.layers[-self.trainable_layer_count:]:
                layer.trainable = True

        # zusätzliche Layer am Schluss einfügen
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dropout(0.5)(x)
        final_output = Dense(1, activation='sigmoid', name='final_output')(x)
        model = Model(input_tensor, final_output)

        return model
