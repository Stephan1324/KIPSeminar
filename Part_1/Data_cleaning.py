import os
import cv2
import shutil
import numpy as np

from CONFIG_GLOBAL import CONFIG_GLOBAL


class Data_cleaning:


    @classmethod
    def clean_data(cls, key_folder) -> None:
        print('------ Start Data Cleaning: ------')

        image_list = Data_cleaning.make_image_list(CONFIG_GLOBAL.PATH_DICT_ORIGINAL[key_folder])
        images_of_same_dimension, images_of_different_dimension = Data_cleaning.get_image_dimension(list_image_names=image_list, key_folder=key_folder)

        print('\n   -------------------------------------')
        print(" Images of same dimension : ", '\n    ', str(images_of_same_dimension))
        print(" Images of different dimension : ", '\n    ', str(images_of_different_dimension))

        Data_cleaning.copy_files_from_to(images_of_same_dimension, CONFIG_GLOBAL.PATH_DICT_ORIGINAL[key_folder],
                                         CONFIG_GLOBAL.PATH_DICT_CLEANED[key_folder])
        return None

    @classmethod
    def make_image_list(cls, path: str = CONFIG_GLOBAL.PATH_ORIGINAL_DATA_FOLDER) -> list:
        file_list = os.listdir(path)
        print('\n   -------------------------------------')
        print(' List of all images in the original_data folder:')
        print('     ', file_list)
        return file_list

    @classmethod
    def read_image_dimension(cls, image_name: list, path: str =CONFIG_GLOBAL.PATH_ORIGINAL_DATA_FOLDER):
        img = cv2.imread(path + image_name, cv2.IMREAD_COLOR)
        dimension = img.shape
        print('     ', dimension)
        return dimension

    @classmethod
    def get_image_dimension(cls,key_folder, list_image_names: list, length: int = 150, width: int = 150,
                            channels: int = 3) -> tuple:
        arr = np.zeros((length, width, channels))
        images_of_same_dimension = []
        images_of_different_dimension = []
        for x in list_image_names:
            if Data_cleaning.read_image_dimension(image_name=x,path= CONFIG_GLOBAL.PATH_DICT_ORIGINAL[key_folder]) == arr.shape:
                images_of_same_dimension.append(x)
            else:
                images_of_different_dimension.append(x)
        return images_of_same_dimension, images_of_different_dimension

    @classmethod
    def copy_files_from_to(cls, files_to_copy: list, src_folder: str, dest_folder: str) -> None:
        print('\n   -------------------------------------')
        print(' Copy files of the same Size to cleaned data folder:')
        for file_name in files_to_copy:
            src_file = os.path.join(src_folder, file_name)
            dest_file = os.path.join(dest_folder, file_name)
            if os.path.exists(dest_file):
                print('     File', file_name, 'already exists!')
            else:
                shutil.copy(src_file, dest_file)
                print('     File', file_name, ' has been copied from', 'to', dest_folder)
        return None
