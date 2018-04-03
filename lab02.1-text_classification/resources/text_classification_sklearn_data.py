from azureml.logging import get_azureml_logger
from sklearn.linear_model import LogisticRegression
import tatk
from tatk.pipelines.text_classification.text_classifier import TextClassifier
from tatk.utils import download_blob_from_storage, resources_dir, data_dir
import pandas as pd
import numpy as np
import math
import os

logger = get_azureml_logger()

if __name__ == '__main__':
    # Data Download
    # -------------
    # Download training and test data from blob strage. 
    # To use your own blob storage update the following parameters:
    #      #connection_string=None, (replace None with your connection string)
    #      #container_name=None,    (replace None with your container name)
    #      blob_name=os.path.join("sentiment", "SemEval2013.Train.tsv") (replace the sub directoy "sentiment" and the file name)

    # Set the working directory where to save the training data files
    resources_dir = "C:\\tatk\\resources"
    download_blob_from_storage(download_dir=resources_dir, 
                            #connection_string=None, 
                            #container_name=None,
                                blob_name=os.path.join("sentiment", "SemEval2013.Train.tsv"))

    download_blob_from_storage(download_dir=resources_dir, 
                                blob_name=os.path.join("sentiment", "SemEval2013.Test.tsv"))

    # Training Dataset Location
    file_path = os.path.join(resources_dir, "sentiment", "SemEval2013.Train.tsv")

    df_train = pd.read_csv(file_path,
                            sep = '\t',                        
                            header = 0, names= ["id","text","label"])

    # Test Dataset Location
    df_test = pd.read_csv(os.path.join(resources_dir,"sentiment", "SemEval2013.Test.tsv"),
                            sep = '\t',                        
                            header = 0, names= ["id","text","label"])

    # Model Training
    # --------------

    log_reg_learner =  LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                                C=1.0, fit_intercept=True, intercept_scaling=1, 
                                class_weight=None, random_state=None, 
                                solver='lbfgs', max_iter=100, multi_class='ovr',
                                verbose=1, warm_start=False, n_jobs=3) 

    text_classifier = TextClassifier(estimator=log_reg_learner, 
                                    text_cols = ["text"], 
                                    label_cols = ["label"], 
                                    extract_word_ngrams=True, extract_char_ngrams=True)


    # Train the model using the default parameters of the package
    text_classifier.fit(df_train)        

    # Performance evaluation

    df_test = text_classifier.predict(df_test)
    text_classifier.evaluate(df_test)