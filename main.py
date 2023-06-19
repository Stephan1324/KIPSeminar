from Model.baseline import Baseline
# from Data_Preparation.Data_cleaning import Data_cleaning
from Deployment.video_deployment import VideoDeployment

if __name__ == "__main__":
    # Data Cleaning für pitting und no_defect Datensätze
    # pitting_preperation = Data_cleaning('pitting')
    # no_detect_preperation = Data_cleaning('no_defect')

    # pitting_preperation.delete_cleaned_data()
    # no_detect_preperation.delete_cleaned_data()
    # pitting_preperation.copy_clean_data()
    # no_detect_preperation.copy_clean_data()

    # erster Ansatz mit baseline_model zum Trainieren
    # mit model_type='resnet_50' kann weiteres model aus dem internet
    # trainiert werden

    model_instance = Baseline(epochs=10,
                              batch_size=16, initial_learningrate=0.01)

    x_images, labels = model_instance.load_classes()

    x_images = model_instance.normalize_data(x_images)

    model_instance.split(x_images, labels)

    model_instance.model_building()

    # model_instance.grid_search(augmentation_factor=3, epochs=[5, 10],
    #                            batch_size=[32, 16],
    #                            learning_rates=[0.001, 0.01])

    # model_instance.fit(augmentation=True, augmentation_factor=4)
    # model_instance.evaluate()
    # model_instance.predict_image(img_number=60, hsv=True)

    # deployment auf dem Video
    deployment = VideoDeployment(
        model_type='baseline', epochs=10, batch_size=16, learning_rate=0.01)
    deployment.predict(normalize=True, hsv=True)
