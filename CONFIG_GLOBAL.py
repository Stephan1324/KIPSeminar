import os


class CONFIG_GLOBAL:
    dirname = os.path.dirname(__file__)
    PATH_ORIGINAL_DATA_FOLDER = os.path.join(dirname, 'original_data')
    PATH_CLEANED_DATA_FOLDER = os.path.join(dirname, 'cleaned_data')

    PATH_ORIGINAL_DATA_NO_DEFECT_SIMPLIFIED_FOLDER = os.path.join(
        dirname, 'original_data', 'KGT_noDefect_simplified')
    PATH_ORIGINAL_DATA_PITTING_SIMPLIFIED_FOLDER = os.path.join(
        dirname, 'original_data', 'KGT_pitting_simplified')
    PATH_CLEANED_DATA_NO_DEFECT_SIMPLIFIED_FOLDER = os.path.join(
        dirname, 'cleaned_data', 'KGT_noDefect_simplified')
    PATH_CLEANED_DATA_PITTING_SIMPLIFIED_FOLDER = os.path.join(
        dirname, 'cleaned_data', 'KGT_pitting_simplified')

    PATH_ORIGINAL_DATA_NO_DEFECT_FOLDER = os.path.join(
        dirname, 'original_data', 'KGT_noDefect')
    PATH_ORIGINAL_DATA_PITTING_FOLDER = os.path.join(
        dirname, 'original_data', 'KGT_pitting')
    PATH_CLEANED_DATA_NO_DEFECT_FOLDER = os.path.join(
        dirname, 'cleaned_data', 'KGT_noDefect')
    PATH_CLEANED_DATA_PITTING_FOLDER = os.path.join(
        dirname, 'cleaned_data', 'KGT_pitting')

    PATH_MODELS_FOLDER = os.path.join(dirname, 'models')
    PATH_DEPLOYMENT_FOLDER = os.path.join(dirname, 'deployment')
    PATH_VIDEO = os.path.join(PATH_DEPLOYMENT_FOLDER, 'pitting_video.mp4')

    PATH_DICT_ORIGINAL = {'KGT_noDefect_simplified':
                          PATH_ORIGINAL_DATA_NO_DEFECT_SIMPLIFIED_FOLDER,
                          'KGT_pitting_simplified':
                          PATH_ORIGINAL_DATA_PITTING_SIMPLIFIED_FOLDER,
                          'KGT_noDefect':
                          PATH_ORIGINAL_DATA_NO_DEFECT_FOLDER,
                          'KGT_pitting':
                          PATH_ORIGINAL_DATA_PITTING_FOLDER
                          }

    PATH_DICT_CLEANED = {'KGT_noDefect_simplified':
                         PATH_CLEANED_DATA_NO_DEFECT_SIMPLIFIED_FOLDER,
                         'KGT_pitting_simplified':
                         PATH_CLEANED_DATA_PITTING_SIMPLIFIED_FOLDER,
                         'KGT_noDefect':
                         PATH_CLEANED_DATA_NO_DEFECT_FOLDER,
                         'KGT_pitting':
                         PATH_CLEANED_DATA_PITTING_FOLDER
                         }
