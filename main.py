from Model.Model import Model

if __name__ == "__main__":
    # Data Cleaning für pitting und no_defect Datensätze
    # Data_cleaning.clean_data('pitting')
    # Data_cleaning.clean_data('no_defect')

    # erster Ansatz mit schlechtem Modell zum trainieren
    model_instance = Model(model_type='baseline')
    model_instance.split(test_size=0.2)
    model_instance.model_building()
    model_instance.fit()
    model_instance.evaluate()
    # model_instance.predict(img_path=r'/Users/stephanernstberger/PycharmProjects/KIPSeminar/cleaned_data/KGT_pitting_simplified/P (60).png')
