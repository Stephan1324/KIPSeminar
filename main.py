from Part_1.Data_utils import Data_utils


if __name__ == "__main__":
    image_list = Data_utils.make_image_list()
    images_of_same_dimension, images_of_different_dimension = Data_utils.get_image_dimension(image_list)
    print("Images of same dimension : " + str(images_of_same_dimension))
    print("Images of differnet dimension : " + str(images_of_different_dimension))

