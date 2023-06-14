# from Model.Model import Model
# from Deployment.Deployment import Video_Deployment
from Data_Preparation.Data_cleaning import Data_cleaning

if __name__ == "__main__":
    # Data Cleaning für pitting und no_defect Datensätze
    pitting_preperation = Data_cleaning('pitting')
    no_detect_preperation = Data_cleaning('no_defect')

    pitting_preperation.delete_cleaned_data()
    no_detect_preperation.delete_cleaned_data()
    pitting_preperation.copy_clean_data()
    no_detect_preperation.copy_clean_data()

    # erster Ansatz mit baseline_model zum trainieren
    # mit model_type='resnet_50' kann weiteres model aus dem internet trainiert werden

    # model_instance = Model(model_type='baseline', epochs=10, batch_size=16, initial_learningrate=0.01)
    # model_instance.split(test_size=0.2, normalize=True, hsv=True)
    # model_instance.model_building()
    # # model_instance.grid_search(augmentation_factor=3, epochs=[5, 10], batch_size=[32, 16], learning_rates=[0.001, 0.01])
    # model_instance.fit(augmentation=True, augmentation_factor=3)
    # model_instance.evaluate()
    # model_instance.predict_image(img_number=60, hsv=True)

    # # deployment auf dem Video
    # deployment = Video_Deployment(model_type='baseline')
    # deployment.predict(normalize=True, hsv=True)
