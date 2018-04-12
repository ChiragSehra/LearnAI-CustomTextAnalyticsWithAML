import os
import pandas as pd
import sys
from tatk.utils import load_biomedical_data, download_embedding_model, data_dir, dictionaries_dir, models_dir
from tatk.pipelines.entity_extraction.keras_entity_extractor import KerasEntityExtractor
from azureml.logging import get_azureml_logger
logger = get_azureml_logger()

if __name__ == "__main__":
    X_train, y_train = load_biomedical_data(data_dir, subset='train', categories='BC5')
    X_test, y_test = load_biomedical_data(data_dir, subset='test', categories='BC5')

    # Create Pandas Dataframes as input into the package functions
    df_train = pd.DataFrame({"text":X_train, "label":y_train})
    df_test = pd.DataFrame({"text":X_test, "label":y_test})

    pubmed_embedding_file_path = download_embedding_model(models_dir, embedding_type='pubmed')
    binary_format = False
    limit = None

    keras_entity_recognizer = KerasEntityExtractor(embedding_file_path=pubmed_embedding_file_path, 
                                                input_col="text", label_col="label", 
                                                prediction_col ="prediction", probabilities_col = "probabilities",
                                                binary_format=binary_format, limit=limit)

    # Export parameters to file
    params_file_path = os.path.join(data_dir, "entity_extraction_params.tsv")
    keras_entity_recognizer.export_params(params_file_path)

    keras_entity_recognizer.set_step_params_by_name("learner", 
                                                    model__recurrence_type = 'LSTM', 
                                                    model__bidirectional = True, 
                                                    batch_size = 50, 
                                                    n_epochs = 5)

    keras_entity_recognizer.fit(df_train)

    df_test_results = keras_entity_recognizer.predict(df_test)
    print(df_test_results.head())

    res = keras_entity_recognizer.evaluate(df_test)
    print(res[['Recall', 'Prec.', 'F-score']])