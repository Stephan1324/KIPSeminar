from CONFIG_GLOBAL import CONFIG_GLOBAL
from Part_1.Data_cleaning import Data_cleaning
from Model.Model import Model

if __name__ == "__main__":
    # Data Cleaning für pitting und no_defect Datensätze
    Data_cleaning.clean_data('pitting')
    Data_cleaning.clean_data('no_defect')


    # erster Ansatz mit schlechtem Modell zum trainieren
    # model_instance = Model()
    # model_instance.split(0.2)
    # model_instance.model_building()
    # model_instance.fit()
    # model_instance.evaluate()