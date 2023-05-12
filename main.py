from Model.Model import Model

if __name__ == "__main__":
    # Data Cleaning für pitting und no_defect Datensätze
    # Data_cleaning.clean_data('pitting')
    # Data_cleaning.clean_data('no_defect')

    # erster Ansatz mit baseline_model zum trainieren
    # mit model_type='model_1' kann weiteres modell aus dem internet trainiert werden
    model_instance = Model(model_type='baseline', epochs=10, batch_size=32)
    model_instance.split(test_size=0.2, normalize=True)
    model_instance.model_building()
    model_instance.fit()
    model_instance.evaluate()
    model_instance.predict(img_number=60)
