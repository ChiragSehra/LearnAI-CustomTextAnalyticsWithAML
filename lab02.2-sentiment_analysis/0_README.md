# Sentiment Analysis using CNN and pre-trained word2Vec model

This hands-on lab demonstrates the use of Convolutional Neural Networks (CNNs) together with a pre-trained embedding model using the AML Package for Text Analytics.

In this lab, we will:
- Utilize pre-trained embedding model that includes word vectors trained on google news for predicting sentiment of movie reviews
- Explore the IMDB dataset by generating summary statistics
- Build a CNN model using TATK's KerasTextClassifier and evaluate the performance of the model
- Save the pipeline, load it and test on unseen reviews to predict sentiment

### Learning Objectives ###

The objectives of this lab are to:

- Understand how you can integrate a pre-trained embedding model in the pipeple
- Understand how to build a CNN model with the pre-trained embedding model for setiment prediction
- Understand how you can use the model for scoring sentiment on unseen reviews

### Sentiment Analysis

Sentiment analysis is a well-known task in the realm of natural language processing (NLP), and it aims to determine the attitude of a speaker/writer. Frequently, artificial neural networks (and deep learning) are used to estimate such sentiment. In this lab, we will learn how to build a CNN model with the pre-trained embedding model for setiment prediction.

### Execution

Launch Azure Machine Learning Workbench and open CLI by selecting File -> Open Command Prompt. Execute the script related to sentiment analysis by runing the below command and walk through the code:

```az ml experiment submit -c local sentiment_analysis.py```

### Data

In this lab, we will load the IMDB dataset using the helper function __load_imdb_data__ as shown below. The helper function is used to download the raw dataset from the URL, and transform it into a pandas DataFrame.

````python
    X_train, y_train, X_test, y_test = load_imdb_data(data_dir)
````

Similarly, the helper function __download_embedding_model__ is used to download the word2Vec word embedding model.

````python
    embedding_file_path = download_embedding_model(models_dir, embedding_type='google')
````

The generic word embedding model has 3M words. We load the top <max_features> words as input features to the neural network with convolutions and pooling operations. The main idea behind a convolution and pooling architecture for language tasks is to apply a learned nonlinear function over each instantiation of a sliding window of k-words over the sentence.

### Training the model

Perform model training by running the below lines. 

````python
keras_text_classifier = KerasTextClassifier(embedding_file_path, 
                                            input_col="review", 
                                            label_col="sentiment",
                                            model_type="convolution",
                                            binary_format=True, 
                                            limit=max_features, 
                                            callbacks=False)

model = keras_text_classifier.fit(df_train)
````

We are only using a slice of the full word embedding matrix for efficiency. During training, you will see information about the different layers and epochs as shown below. One epoch consists of one full training cycle on the training set. 


|Layer (type)     | Output Shape     | Param #   
|-----------------|------------------|--------
|embedding_1 (Embedding)|      (None, None, 300) |        30000600|
|dropout_1 (Dropout)    |      (None, None, 300) |        0       ||
|conv1d_1 (Conv1D)      |      (None, None, 250) |        225250  |

````
Epoch 1/5
500/500 [==============================] - 63s 125ms/step - loss: 0.3761 - acc: 0.8248
Epoch 2/5
500/500 [==============================] - 36s 71ms/step - loss: 0.2606 - acc: 0.8937
Epoch 3/5
500/500 [==============================] - 30s 61ms/step - loss: 0.2108 - acc: 0.9156
Epoch 4/5
500/500 [==============================] - 29s 59ms/step - loss: 0.1761 - acc: 0.9310
Epoch 5/5
500/500 [==============================] - 28s 57ms/step - loss: 0.1379 - acc: 0.9462
````


### Evaluation

The text classifier can be applied on the test data frame using the __predict__ function as follows to get the predictions: 

````python
keras_text_classifier.predict(df_test)
````

|review|sentiment|prediction|
|------|---------|----------|
WOW! What a horrible, hideous waste of time th...|0|0
Enjoyed this film which deals entirely about a...|1|1
This movie literally had me rolling on the flo...|1|1
Critters 4 starts, & I quote 'Somewhere in Kan...|0|0
First off, I'm not a firefighter, but I'm in s...|0|0

Obtaining evaluation metrics is as easy as calling the __evaluate__ function. For example, the below line will produce confusion matrix and macro-f1:

````python
res = keras_text_classifier.evaluate(df_test)
````

__plot_confusion_matrix__ function can also be used to generate normalized confusion matrix by setting ````normalize=True````. You will be able to see the normalized cells in the matrix as follows along with the raw matrix at the end of script execution.

````
Confusion matrix
[[11454  1073]
 [ 1286 11187]]

Normalized confusion matrix
[[0.91434501 0.08565499]
 [0.1031027  0.8968973 ]]
````

## Exercise (Optional)

1. Modify the number of features used in model training and notice the difference in performance with varying number of features. Can you plot a graph of macro-f1 measure against the number of features?

2. What are some potential limitations of the task of sentiment analysis?


