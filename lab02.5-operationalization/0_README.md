# Operationalization

This hands-on lab demonstrates the application of the Azure ML Text Analytics Package for operationalization of sentiment prediction.

In this lab, we will:
- Train and deploy a text classification model
- Use the service endpoint for predicting the sentiment of new text

### Learning Objectives ###

The objectives of this lab are to:
- Learn how to setup your environment for deploying
- Learn how to create a service endpoint by deploying a trained model
- Use the service endpoint for sentiment prediction

### Execution

For deployment, we would need to install the following wheel files using pip from CLI:

```
pip install https://aztatksa.blob.core.windows.net/tatk-build-drops/PullRequest/azureml_tatk-0.1.18113.15a1-py3-none-any.whl

pip install https://azuremlftkrelease.blob.core.windows.net/latest/azuremltkbase-1.0.0b1-py3-none-any.whl 
```

To execute the scoring script `train_deploy.py` in the resources folder, you can run the following command from CLI:

`az ml experiment submit -c local train_deploy.py`

However, to run the script and deploy, there are few things to configure and change in the script. We will review the script and configuration files in the following sections.

### Data

The script downloads sentiment datasets (_SemEval2013.Train.tsv_ and _SemEval2013.Train.tsv_) from blob storage using the helper function _download_blob_from_storage_. A sample of the dataset is as follows. The text is associated with a sentiment label (positive, neutral and negative):

|id|text|label|
|--|----|-----| 
|1|@killa_1983 - If you ain't doing nothing Satur...|positive|
|2|- Pop bottles , make love , thug passion , RED...|positive|
|3|@TheScript_Danny @thescript - St Patricks Day ...|positive|
|4|@TheScript_Danny @thescript - St Patricks Day ...|positive|
|5|@DJT103 - You know what the holidays alright w...|positive|

### Model Training and Evaluation

This step involves training a Scikit-learn text classification model using One-versus-Rest LogisticRegression learning algorithm. The model is trained on the 'text' field with labels from 'label' field using the default parameters of the package (i.e. extraction of word unigrams and bigrams and character 4-grams).

````
log_reg_learner =  LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                            C=1.0, fit_intercept=True, intercept_scaling=1, 
                            class_weight=None, random_state=None, 
                            solver='lbfgs', max_iter=100, multi_class='ovr',
                            verbose=1, warm_start=False, n_jobs=3) 

text_classifier = TextClassifier(estimator=log_reg_learner, 
                                text_cols = ["text"], 
                                label_cols = ["label"], 
                                extract_word_ngrams=True, extract_char_ngrams=True)
````

`text_classifier.fit(df_train)` performs training. Evaluation is performed on the dataset _df_test_ and after execution of the below snippet, you will see confusion matrix and macro_f1.

````python
    text_classifier.fit(df_train)
    df_test = text_classifier.predict(df_test)
    text_classifier.evaluate(df_test)          
````

We now have a model that we can deploy!

### Deployment Setup

To deploy the trained model to production, there are several setup steps needed using CLI before running `train_deploy.py`.

1. Set up your environment using the following command:

    `az ml env setup --cluster -n <ENVIRONMENT_NAME> -l <AZURE_REGION e.g. eastus2> [-g <RESOURCE_GROUP>]`

    For examle, `az ml env setup --cluster -n sentienv -l eastus2`

2. Ensure that the _provisioning state_ of the environment setup is _Succeeded_. You can check the _provisioning state_ by running the below command:

    `az ml env show -n <ENVIRONMENT_NAME> -g <RESOURCE_GROUP>`

    For example, `az ml env show -g sentienvrg -n sentienv` would produce:

    ```
    {
      "Cluster Name": "sentienv",
      "Cluster Size": 2,
      "Created On": "2018-04-26T06:00:47.45799999999999996Z",
      "Location": "eastus2",
      "Provisioning State": "Succeeded",
      "Resource Group": "sentienvrg",
      "Subscription": "5be49961-ea44-42ec-8021-b728be90d58c"
    }
    ```

3. Once _provisioning state_ changes from _Creating_ to _Succeeded_, we can set the above environment as our compute environment:

    `az ml env set -n <ENVIRONMENT_NAME> -g <RESOURCE_GROUP>`

    For exampe, `az ml env set -g sentienvrg -n sentienv`

4. A model management account is required for deploying models. We usually do this once per subscription, and can reuse the same account in multiple deployments. To create a new model management account and use the model management account, run the below commands:

    `az ml account modelmanagement create -l <AZURE_REGION e.g. eastus2> -n <ACCOUNT_NAME> -g <RESOURCE_GROUP> --sku-instances <NUMBER_OF_INSTANCES, e.g. 1> --sku-name <PRICING_TIER for example S1>`

    For example, `az ml account modelmanagement create -l eastus2 -n sentmodel -g sentienvrg --sku-instances 1 --sku-name S1`

    ```
    az ml account modelmanagement create -l eastus2 -n cvtkmodel -g cvtkevnrg --sku-instances 1 --sku-name S1
    
    az ml account modelmanagement set -n cvtkmodel -g cvtkevnrg
    ```

Make a note of the environment and model information as we will need it in the next section.

### Deploy Configuration

The below code snippet setups deployment using the configuration file `tatk_deploy_config.yaml`. In addition, you will need to provide a _web_service_name_ for the scoring service.

````python
    deployment_config_file_path=os.path.join(resources_dir, )

    web_service_name = 'your service name'
    working_directory= os.path.join(resources_dir, 'deployment') 

    web_service = text_classifier.deploy(web_service_name=web_service_name, 
                        config_file_path=deployment_config_file_path,
                        working_directory= working_directory)

````

The configuration file `tatk_deploy_config.yaml` would need to be modified to capture all the information used in setting up the environment and model management acount in the _Deployment Setup_ section. An example of `tatk_deploy_config.yaml` is below:

```
    azure_subscription: Your Subscription
    env_name: sentienv
    env_location: eastus2
    env_resource_group: sentienvrg
    model_management_account_name: sentmodel
    model_management_account_resource_group: sentienvrg
    model_management_account_location: eastus2
    env_agent_count: 2
    env_agent_vm_size: Standard_D3_v2
    env_master_count: 1
    reuse_storage_and_acr: n
    cluster: cluster
```

    tatk_web_service = CsiWebService(web_service_name)

You can also get web_service _url_, _api_key_ and _id_ as follows and obtain usage information using _CsiWebService_.

````python
    print("Service URL: {}".format(web_service._service_url))
    print("Service URL: {}".format(web_service._api_key))
    print("Service Id: {}".format(web_service._id))

    tatk_web_service = CsiWebService(web_service_name)
````
### Scoring Service

1. To run the service from CLI, you can get the Usage information for `Sample CLI command' when you run the script. An example to score using CLI is as follows:

    ```
    az ml service run realtime -i sentservice.sentienv-c6371181.eastus2 -d "{\"values\": [{\"recordId\": \"0\", \"data\": {\"text\": \"@caplannfl - Another example of a good college player who had a great week at Senior Bowl to ease concerns about toughs and get into 1st round\"}}, {\"recordId\": \"1\", \"data\": {\"text\": \"- Grassroots racing at its best returns to Marana on Saturday, October 27, at the Gladden Farms 10K an http://t.co/czsOxWoQ\"}}]}"
    ```

    The scored result captures the sentiment class for the corresponding text:

    ````json
    {'values': [{'recordId': '0', 'data': {'class': 'positive', 'text': '@caplannfl - Another example of a good college player who had a great week at Senior Bowl to ease concerns about toughs and get into 1st round'}}, {'recordId': '1', 'data': {'class': 'positive', 'text': '- Grassroots racing at its best returns to Marana on Saturday, October 27, at the Gladden Farms 10K an http://t.co/czsOxWoQ'}}]}
    ````

2. You can also pass text via script and get it scored using the web service as follows:

````python
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
````

Below is an example of text that is scored:

````json
{"values": [{"recordId": "a1", "data": {"text": "a good college player who had a great week"}}, {"recordId": "b2", "data": {"text": "a bad college player who had a awful week"}}]}
F1 2018-04-27 08:46:22,593 INFO Web service scored.
----------------Prediction-----------------
{"values": [{"recordId": "b2", "data": {"class": "neutral", "text": "a bad college player who had a awful week"}}, {"recordId": "a1", "data": {"class": "positive", "text": "a good college player who had a great week"}}]}
````

2. Additionally, you can also use the web service url and keys to call the service within your code.