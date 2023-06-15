import os
import shutil
import numpy as np
import cv2

from CONFIG_GLOBAL import CONFIG_GLOBAL


class Data_cleaning:

    def __init__(self, key_folder, length: int = 150, width: int = 150,
                 channels: int = 3):
        """Copies the files which correstpond to the given setting into the
        corresponding cleaned data folder

        Args:
        key_folder: Either 'pitting' or 'no_defect can be picked
        length(int): Number of pixles
        width(int): Number of pixles
        channels(int): Number of channels
        """

        self.origin_data_path = CONFIG_GLOBAL.PATH_DICT_ORIGINAL[key_folder]
        self.cleaned_data_path = CONFIG_GLOBAL.PATH_DICT_CLEANED[key_folder]
        self.image_list = os.listdir(self.origin_data_path)
        self.image_standards = np.zeros((length, width, channels))
        self.seperate_image_list = self.set_seperate_image_list()

    def set_seperate_image_list(self):
        """
        Returns:
            seperate_image_list: List with corresponing and non-corresponding
            images
        """
        images_of_same_dimension = []
        images_of_different_dimension = []

        for image in self.image_list:
            image_path = os.path.join(self.origin_data_path, image)
            image_cv2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_dimension = image_cv2.shape

            if image_dimension == self.image_standards.shape:
                images_of_same_dimension.append(image)
            else:
                images_of_different_dimension.append(image)

        seperate_image_list = [images_of_same_dimension,
                               images_of_different_dimension]

        return seperate_image_list

    def delete_cleaned_data(self):

        # Empties the currenct cleaned_data folder
        for file_name in os.listdir(self.cleaned_data_path):
            file_path = os.path.join(self.cleaned_data_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def copy_clean_data(self):

        for file_name in self.seperate_image_list[0]:
            source_file = os.path.join(self.origin_data_path, file_name)
            destination_file = os.path.join(self.cleaned_data_path, file_name)

            if not os.path.exists(destination_file):
                shutil.copy(source_file, destination_file)
