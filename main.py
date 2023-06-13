from Model.Model import Model
from Deployment.Deployment import Video_Deployment

if __name__ == "__main__":
    # Data Cleaning für pitting und no_defect Datensätze
    # Data_cleaning.clean_data('pitting')
    # Data_cleaning.clean_data('no_defect')

    # erster Ansatz mit baseline_model zum trainieren
    # mit model_type='resnet_50' kann weiteres model aus dem internet trainiert werden

    model_instance = Model(model_type='baseline', epochs=1, batch_size=32)
    model_instance.split(test_size=0.2, normalize=True, hsv=True)
    model_instance.model_building()
    # model_instance.grid_search(epochs=[1, 2], batch_size=[32])
    model_instance.fit(augmentation=True, augmentation_factor=4)
    model_instance.evaluate()
    model_instance.predict_image(img_number=60, hsv=True)


    # deployment auf dem Video
    deployment = Video_Deployment(model_type='baseline')
    deployment.predict(normalize=True, hsv=True)
