import os


class CONFIG_GLOBAL:
    dirname = os.path.dirname(__file__)
    PATH_ORIGINAL_DATA_FOLDER = os.path.join(dirname, 'original_data')
    PATH_CLEANED_DATA_FOLDER = os.path.join(dirname, 'cleaned_data')

    PATH_ORIGINAL_DATA_NO_DEFECT_FOLDER = os.path.join(
        dirname, 'original_data', 'KGT_noDefect_simplified')
    PATH_ORIGINAL_DATA_PITTING_FOLDER = os.path.join(
        dirname, 'original_data', 'KGT_pitting_simplified')
    PATH_CLEANED_DATA_NO_DEFECT_FOLDER = os.path.join(
        dirname, 'cleaned_data', 'KGT_noDefect_simplified')
    PATH_CLEANED_DATA_PITTING_FOLDER = os.path.join(
        dirname, 'cleaned_data', 'KGT_pitting_simplified')

    PATH_MODEL_FOLDER = os.path.join(dirname, 'Model')
    PATH_DEPLOYMENT_FOLDER = os.path.join(dirname, 'Deployment')
    PATH_VIDEO = os.path.join(PATH_DEPLOYMENT_FOLDER, 'pitting_video.mp4')

    PATH_DICT_ORIGINAL = {'no_defect': PATH_ORIGINAL_DATA_NO_DEFECT_FOLDER,
                          'pitting': PATH_ORIGINAL_DATA_PITTING_FOLDER}

    PATH_DICT_CLEANED = {'no_defect': PATH_CLEANED_DATA_NO_DEFECT_FOLDER,
                         'pitting': PATH_CLEANED_DATA_PITTING_FOLDER}
