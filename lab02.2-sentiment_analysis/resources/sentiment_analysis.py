import numpy as np
import pandas as pd
import os
import keras
import itertools
from scipy import stats
import pandas as pd
from sklearn.metrics import confusion_matrix
from tatk.utils.load_data import load_imdb_data, download_embedding_model, data_dir, models_dir
from tatk.pipelines.text_classification.keras_text_classifier import KerasTextClassifier
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

if __name__ == "__main__":

    # ## Step 1: Loading dataset and pre-trained embedding model
    X_train, y_train, X_test, y_test = load_imdb_data(data_dir)

    df_train = pd.DataFrame({'review': X_train, 'sentiment': y_train})
    df_test = pd.DataFrame({'review': X_test, 'sentiment': y_test})
    embedding_file_path = download_embedding_model(models_dir, embedding_type='google')

    # ## Step 2: Model training
    # the generic word embedding model has 3M words. We load the top <max_features> words as input features to the neural network
    max_features=100000

    keras_text_classifier = KerasTextClassifier(embedding_file_path, 
                                                input_col="review", 
                                                label_col="sentiment",
                                                model_type="convolution",
                                                binary_format=True, 
                                                limit=max_features, 
                                                callbacks=False)

    keras_text_classifier.set_step_params_by_name("vectorizer", get_from_path=False)
    model = keras_text_classifier.fit(df_train)


    # ## Step 3: Apply the text classifier

    keras_text_classifier.predict(df_test)
    print(df_test.head())

    res = keras_text_classifier.evaluate(df_test)

    cnf_matrix = confusion_matrix(y_pred=df_test['prediction'].values, y_true=df_test['sentiment'].values)
    class_labels = ['pos', 'neg']
    print_confusion_matrix(cnf_matrix, classes = class_labels, normalize=True,
                        title='Normalized confusion matrix')