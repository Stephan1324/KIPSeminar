from data_cleaning.data_cleaning import DataCleaning

if __name__ == "__main__":
    # Data Cleaning für pitting- und no_defect- simppflified Datensätze
    pitting_preperation = DataCleaning('KGT_pitting_simplified')
    noDefect_preperation = DataCleaning('KGT_noDefect_simplified')

    pitting_preperation.delete_cleaned_data()
    noDefect_preperation.delete_cleaned_data()
    pitting_preperation.copy_clean_data()
    noDefect_preperation.copy_clean_data()

    # # Data Cleaning für pitting und no_defect Datensätze
    # pitting_preperation = DataCleaning('pitting')
    # noDefect_preperation = DataCleaning('no_defect')

    # pitting_preperation.delete_cleaned_data()
    # noDefect_preperation.delete_cleaned_data()
    # pitting_preperation.copy_clean_data()
    # noDefect_preperation.copy_clean_data()
