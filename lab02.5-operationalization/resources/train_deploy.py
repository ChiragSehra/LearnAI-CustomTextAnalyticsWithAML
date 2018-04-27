from azureml.logging import get_azureml_logger
from tatk.utils import download_blob_from_storage, resources_dir, data_dir
from sklearn.linear_model import LogisticRegression
from tatk.pipelines.text_classification.text_classifier import TextClassifier
from tatk.operationalization.csi.csi_web_service import CsiWebService
import os
import tatk
import pandas as pd
import numpy as np
import math
import json

logger = get_azureml_logger()

if __name__ == "__main__":


    #set the working directory where to save the training data files
    resources_dir = os.path.join(os.path.expanduser("~"), "tatk", "resources")

    download_blob_from_storage(download_dir=resources_dir, 
                                blob_name=os.path.join("sentiment", "SemEval2013.Train.tsv"))
            
    download_blob_from_storage(download_dir=resources_dir, 
                                blob_name=os.path.join("sentiment", "SemEval2013.Test.tsv"))

    # Training Dataset Location
    file_path = os.path.join(resources_dir, "sentiment", "SemEval2013.Train.tsv")

    df_train = pd.read_csv(file_path,
                            sep = '\t',                        
                            header = 0, names= ["id","text","label"])
    df_train.head()

    # Test Dataset Location
    df_test = pd.read_csv(os.path.join(resources_dir,"sentiment", "SemEval2013.Test.tsv"),
                            sep = '\t',                        
                            header = 0, names= ["id","text","label"])

    data = df_train["label"].values
    labels = set(data)
    bins = range(len(labels)+1) 

    data = df_test["label"].values
    labels = set(data)
    bins = range(len(labels)+1) 

    log_reg_learner =  LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                                C=1.0, fit_intercept=True, intercept_scaling=1, 
                                class_weight=None, random_state=None, 
                                solver='lbfgs', max_iter=100, multi_class='ovr',
                                verbose=1, warm_start=False, n_jobs=3) 

    text_classifier = TextClassifier(estimator=log_reg_learner, 
                                    text_cols = ["text"], 
                                    label_cols = ["label"], 
                                    extract_word_ngrams=True, extract_char_ngrams=True)

    text_classifier.fit(df_train)
    df_test = text_classifier.predict(df_test)
    text_classifier.evaluate(df_test)          

    deployment_config_file_path=os.path.join(resources_dir, 'tatk_deploy_config.yaml')
    print("resources_dir:", resources_dir)

    web_service_name = 'senservice'
    working_directory= os.path.join(resources_dir, 'deployment') 

    web_service = text_classifier.deploy(web_service_name= web_service_name, 
                        config_file_path=deployment_config_file_path,
                        working_directory= working_directory)  

    print("Service URL: {}".format(web_service._service_url))
    print("Service URL: {}".format(web_service._api_key))
    print("Service Id: {}".format(web_service._id))

    tatk_web_service = CsiWebService(web_service_name)

    dict1 ={}
    dict1["recordId"] = "a1" 
    dict1["data"]= {}
    dict1["data"]["text"] = "a good college player who had a great week"

    dict2 ={}
    dict2["recordId"] = "b2"
    dict2["data"] ={}
    dict2["data"]["text"] = "a bad college player who had a awful week"
    dict_list =[dict1, dict2]
    data ={}
    data["values"] = dict_list
    input_data_json_str = json.dumps(data)
    print (input_data_json_str)
    prediction = tatk_web_service.score(input_data_json_str)
    print("----------------Prediction-----------------")
    print (prediction)