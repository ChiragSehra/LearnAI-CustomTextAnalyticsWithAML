# Entity Extraction

This hands-on lab demonstrates how you can train a custom entity extraction model using TATK.

In this lab, we will:
- Load benchmark biomedical datasets for extracting biomedical entities
- Utilize a word2Vec embedding model to define the entity extractor pipeline
- Save and load the scoring pipeline on unseen data
- Compare with generic entity extractors

### Learning Objectives ###

The objectives of this lab are to:

- Understand entity extraction applied on Pubmed abstracts
- Understand how to develop entity extractor pipeline and evaluate the performance of the pipeline
- Demonstrate that domain-specific word embeddings model can outperform generic word embeddings models

## Entity Extraction

In this lab, we will build an entity extractor that uses APIs that call Keras wtih Tensorflow backend for training and prediction.

In an entity recognition task, each word is mapped to a tag. A tag can have one of the values:

- `B-<entity_type>` -- beginning of an entity phrase of type `<entity_type>` <br>
- `I-<entity_type>` -- non-initial part of an entity phrase of type `<entity_type>` <br>
- `O` -- word outside of any phrase of interest. <br>

`<entity_type>` is one of the entity types of interest. In this example, we use Semeval BC5 dataset with two entity types: *Disease* and *Chemical*. Therefore, allowed tags are: <br>
`B-Disease`, `I-Disease`, `B-Chemical`, `I-Chemical` and `O`. <br>

### Execution

1. Launch Azure Machine Learning Workbench and open CLI by selecting File -> Open Command Prompt.

2. Execute the entity extraction related script by runing the below command and walk through the code:

```az ml experiment submit -c local custom_entity_extraction.py```

## Dataset and Models

The Semeval BC5 dataset is obtained using _load_biomedical_data_ where each sentence is tokenized and annotated. For example, the below annotated sentence

\<B-Chemical\>Naloxone\<B-Chemical\> reverses the antihypertensive effect of \<B-Chemical\>clonidine\<B-Chemical\>.

is transposed into the following format (x, y) for processing where x contains tokenized sentences  and y contains annotations:

['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']

A common scenario is to also extract generic entities such as people's names, locations, organizations, etc. There are several pre-trained models (from Stanford NLP, OpenNLP, etc.) for these purposes. You can create your own annotated data using the same convention. For example:

['Roger', 'is', 'from', 'Seattle'], ['People-Names', 'O', 'O', 'Location']


In addition, we will also use pre-trained embeddings for the pubmed domain as shown below:

````python
pubmed_embedding_file_path = download_embedding_model(models_dir, embedding_type='pubmed')
binary_format = False
limit = None
````

In scenarios where you do not have a domain-specific model, you can use a generic word2Vec embeddings binary file. For example:

````python
google_embedding_file_path = download_embedding_model(models_dir, embedding_type='google')
````

## Model Training

An entity extraction pipeline is implemented in the class `KerasEntityExtractor`. It covers the following main steps:
- `NltkPreprocessor`: to tokenize sentences.
- `Word2VecVectorizer`: to assign each word its embedding feature (from the pre-trained Word2Vec model). 
- `KerasEmbeddingSequenceTagger`: pads the features and train an RNN model using Keras with Tensorflow backend.

The code snippet to build the keras entity extractor is as follows:

````python
from tatk.pipelines.entity_extraction.keras_entity_extractor import KerasEntityExtractor

keras_entity_recognizer = KerasEntityExtractor(embedding_file_path=pubmed_embedding_file_path, input_col="text", label_col="label", prediction_col ="prediction", probabilities_col = "probabilities", binary_format=binary_format, limit=limit)
````

## Parameters

To view the parameters of `keras_entity_recognizer`, you can also export them to a file by running the below commands:

params_file_path = os.path.join(data_dir, "entity_extraction_params.tsv")
keras_entity_recognizer.export_params(params_file_path)

Open the file containing the parameters and investigate the large number of parameters used:

````
vectorizer__embedding_size	100
vectorizer__negative_sample_size	5
learner__model_fn	<function keras_embedding_sequence_tagger_fn at 0x000001835D0780D0>
nltk_preprocessor__input_col	text
nltk_preprocessor__output_col	NltkPreprocessor70b3387b55f84955995c78bd1c1ccd30
learner__model__dropout_rate	0.2
learner__model__init_wordvecs	[[ 3.48426998e-01 -5.51161003e+00 -1.13765001e+01 ... -9.55230045e+00
   5.18775988e+00 -6.96023989e+00]
 [-2.28482008e+00 -1.17461003e-01 -1.22643006e+00 ... -3.60893989e+00
  -1.03862000e+00  1.48386002e+00]
 ...
````

By now, you should know how to view the parameters using the API. Obtain the parameters of the model by running the below command:

````python
keras_entity_recognizer.get_step_params_by_name("learner")
````

1. What kind of a model is used for entity extraction?

2. Can you change the parameters to use a vanilla uni-directional RNN model for faster training using the following parameters?

- model__recurrence_type = 'RNN'
- model__bidirectional= False
- batch_size = 100
- n_epochs = 5

## Model Fitting

Fit entity extractor model with Keras, Tensorflow backend, and RNN layers by running the below line:

````python
keras_entity_recognizer.fit(df_train)
````

Notice information about the model and it's layers being displayed when you fit the entity extractor model. Additionally, each epoch and its associated accuracy is also displayed. For example:

````
| Layer (type) | Output Shape | Param # |   
|-------------|-------------|-------|
|embedding_2 (Embedding)  |    (None, None, 50)     |     4203300 

Epoch 1/5
186/186 [==============================] - 69s 372ms/step - loss: 0.1826 - acc: 0.9534
Epoch 2/5
186/186 [==============================] - 67s 358ms/step - loss: 0.0717 - acc: 0.9771
Epoch 3/5
186/186 [==============================] - 68s 366ms/step - loss: 0.0586 - acc: 0.9810
Epoch 4/5
186/186 [==============================] - 67s 363ms/step - loss: 0.0517 - acc: 0.9829
Epoch 5/5
186/186 [==============================] - 67s 359ms/step - loss: 0.0469 - acc: 0.9844
````

## Evaluation

The predictions on the test set can be obtained by investigating the __df_test_results__ data frame after running the below line:

````python
df_test_results = keras_entity_recognizer.predict(df_test)
````

|text|label|prediction|
|----|-----|----------|
|[Famotidine-associated, delirium, .]|[B-Chemical, B-Disease, O]|[B-Chemical, B-Disease, O]


Obtaining evaluation metrics is as easy as calling the __evaluate__ function. For example, the below lines will produce precision, recall and f-score:

````python
res = keras_entity_recognizer.evaluate(df_test)
print(res[['Recall', 'Prec.', 'F-score']])
````

The below table contains breakdown of evaluation metrics for each entity and all of the entities.

            ||Recall|Prec.|F-score|
            |---|-----|-----|-----|
|Disease|0.643222|0.700318|0.670557|
|Chemical|0.713570|0.832165|0.768318|
|ALL|0.681798|0.770368|0.723382|

## Exercise

1. Can you extract entities on an unseen text? 

    __Hint__: You can create a data frame with the text and pass the data frame in the predict function.

2. Can you compare the performance of different models: LSTM vs RNN? 

3. Are there any significant training time differences between the two models?

4. (OPTIONAL) There are plenty of annotated datasets available at https://www.clips.uantwerpen.be/conll2003/ner/. Can you incorporate the datasets using the approach presented in this lab for Named Entity Extraction?