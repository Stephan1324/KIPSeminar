import os
import cv2
import numpy as np
from CONFIG_GLOBAL import CONFIG_GLOBAL


class Data_utils:

    @classmethod
    def make_image_list(cls, path: str = CONFIG_GLOBAL.PATH_DATA_FOLDER) -> list:
        file_list = os.listdir(path)
        print('List of all images in the data folder:')
        print(file_list)
        return file_list

    @classmethod
    def read_image_dimension(cls, image_name: list):
        img = cv2.imread(CONFIG_GLOBAL.PATH_DATA_FOLDER + image_name, cv2.IMREAD_COLOR)
        dimension = img.shape
        print(dimension)
        return dimension

    # @classmethod
    # def get_image_shapes(cls, list_image_names: list) -> dict:
    #     list_of_dimensions = [Data_utils.read_image_dimension(x) for x in list_image_names]
    #     dimension_dict = dict(zip(list_image_names, list_of_dimensions))
    #     print("Image Dimensions are: " + str(dimension_dict))
    #     return dimension_dict

    @classmethod
    def get_image_dimension(cls, list_image_names: list, length: int = 736, width: int = 560,
                                 channels: int = 3) -> list:
        arr = np.zeros((length, width, channels))
        images_of_same_dimension = []
        images_of_different_dimension = []
        for x in list_image_names:
            if Data_utils.read_image_dimension(x) == arr.shape:
                images_of_same_dimension.append(x)
            else:
                images_of_different_dimension.append(x)

        return images_of_same_dimension, images_of_different_dimension
