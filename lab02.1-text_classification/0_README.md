# Data Ingestion and Text Classification

This hands-on lab demonstrates the application of the Azure ML Text Analytics Package for text classification. 

In this lab, we will:
- Build a simple pipeline by using a Scikit Learner and ingest data
- Set the parameters of pipeline steps
- Evaluate the trained classifier
- Save and load the pipeline for scoring

### Learning Objectives ###

The objectives of this lab are to:

- Understand the text classification workflow
- Get introduced to a text classification pipeline using TATK
- Understand how to ingest datasets into the workflow
- Understand how to perform pre-processing by changing the pipeline's parameters
- Learn how to build the text classifier using different Sckit Learners and perform evaluation

### Datasets

In this lab, we will use two datasets (located at https://aztatksa.blob.core.windows.net/textprivatepreview) for performing text classification.

#### 1. Newsgroups Dataset

The 20 newsgroups collection has become a popular data set for experiments in text mining, such as text classification and text clustering. The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. In this lab, we will be using 4 categories - comp.graphics, rec.autos, sci.med and misc.forsale. You can also opt to load the newsgroup dataset yourself.

#### 2. Sentiment Evaluation Dataset

The sentiment evaluation dataset is used to demonstrate how you can ingest your own dataset to perform text classification. Each record with text is associated with a label (positive/negative).

| id | text | label |
| ----- | ----- | ----- |
| 1 | @killa_1983 - If you ain't doing nothing Satur... | positive |
| 2 | Pop bottles , make love , thug passion , RED... | positive |
| 3 | @TheScript_Danny @thescript - St Patricks Day ... | positive |

### Execution

1. Launch Azure Machine Learning Workbench and open CLI by selecting File -> Open Command Prompt.

2. The two scripts for performing text classification are:
    
    a. ```resources/text_classification_sklearn.py``` (uses newsgroups dataset)
    
    b. ```resources/text_classification_sklearn_data.py``` (uses sentiment evaluation dataset)

Execute the scripts by runing the below command and walk through the code:

```az ml experiment submit -c local <script>```

For example, to execute ```text_classification_sklearn.py```, run the below command:

```az ml experiment submit -c local text_classification_sklearn.py```

### Pipeline Creation

With TATK's pipeline creation, the user does not have to think of composing transformers manually. By default, the pipeline extracts word n-grams and character n-grams. One can also include a rich set of preprocessors and feature extractors to the pipeline including word2vec/fasttext semantic features or thesari lookup.

The below lines in ```text_classification_sklearn.py``` would create a one-versus-rest LogisticRegression learning algorithm that is used in TextClassifier for model training. 

-  njobs=3 in log_reg_learner sets the algorithm to run on 3 threads.

```python
    log_reg_learner =  LogisticRegression(penalty='l2', ..., n_jobs=3)

    text_classifier = TextClassifier(estimator=log_reg_learner, ...)
```

- During pipeline creation, you will see diagnostic messages such as:
````
    TextClassifier::create_pipeline ==> start
    :: number of jobs for the pipeline : 6
    0       text_nltk_preprocessor
    1       text_word_ngrams
    2       text_char_ngrams
    3       assembler
    4       learner
    TextClassifier::create_pipeline ==> end
````
- There are 8 nodes in the pipeline but not all of them run in parallel. Node 1, 2 depend on Node 0. Node 4, 5 depend on Node 3 and Node 7 depends on Node 6.

### Parameters

To read the parameters of the different pipeline steps, you will need to call ```get_step_params_by_name``` and pass the step name of the pipeline as an argument.

```python
text_classifier.get_step_params_by_name("text1_char_ngrams")
```
This will display all parameter values such as:

```json
{'lowercase': True, 'dtype': <class 'numpy.float32'>, 'use_idf': True, 'binary': False, 'input': 'content', 'max_df': 1.0, 'smooth_idf': True, 'input_col': 'NltkPreprocessorfb41531f4098427781f12c99309f6a61', 'tokenizer': None, 'n_hashing_features': None, 'save_overwrite': True, 'vocabulary': None, 'stop_words': None, 'strip_accents': None, 'sublinear_tf': False, 'token_pattern': '(?u)\\b\\w\\w+\\b', 'min_df': 3, 'encoding': 'utf-8', 'norm': None, 'decode_error': 'strict', 'max_features': None, 'ngram_range': (4, 4), 'analyzer': 'char_wb', 'output_col': 'NGramsVectorizer618dec7ba4e14b5099d23132ac2db2e4', 'preprocessor': None, 'hashing': False}
```

You can set the parameters using the ```` set_step_params_by_name```` function as follows:

```python
    text_classifier.set_step_params_by_name("text1_char_ngrams", ngram_range = (3,4), use_idf = False) 
    char_ngrams_params = text_classifier.get_step_params_by_name("text1_char_ngrams")
```

### Artifacts

The script ```text_classification_sklearn.py``` produces three files:

1. ````char_ngrams_vocabulary````: Includes vocabulary of all the character ngrams
2. ````sk_model````: Includes the pipeline pkl files
3. ````word_ngrams_vocabulary````: Includes vocabulary of all the ngram tokens

### Evaluation

When the script finishes running, you will see evaluation measures displayed including confusion matrix and macro f1. For example:

````
[[362   4  17   6]
 [  7 370  12   1]
 [  6   9 380   1]
 [ 17  16  29 334]]

macro_f1 = 0.9203331877515919
````

Micro- and macro-averages (for a given metric) will produce slightly different computations, and thus their interpretation differs. A macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally), whereas a micro-average will aggregate the contributions of all classes to compute the average metric. 

In a multi-class classification setup, micro-average is preferable if you suspect there might be class imbalance (i.e., you may have many more examples of one class than of other classes).

### Exercises

1. There are several pre-processing steps you can perform to improve your model and also reduce the vocabulary space. For example, you can introduce removal of stop words, stemming, etc.

````python
text_classifier.set_step_params_by_name("text1_nltk_preprocessor", remove_stopwords=True)

text_classifier.set_step_params_by_name("text2_nltk_preprocessor", remove_stopwords=True)
````
    Open the word_ngrams_vocabulary file to see if there was a reduction in the vocabulary size. 

    1.1   What percentage is the reduction in the vocabulary space?

    1.2   Is there an improvement in the f1 score after adding the two pre-processing steps?

    1.3   You should not see any stop words in the word_ngrams_vocabulary. However, a few stop-words such as "and" have slipped in. Could you explore why?

    HINT: Investigate the following lines:
````
    " &quot;The universe is a living being, and it's conscious, and it's very old. And it cares about itself in lots of ways.&quot; Drunvalo Melchizedek."
    
    "& quot ; the universe living , ' conscious , ' old . and cares lots ways .& quot ; drunvalo melchizedek .",
````
2. We built a text classifier using sklearn's simple LogisticRegression. Can you include a decision tree learner now and compare the f1 score with the logistic regression model built earlier?

    ```python
    from sklearn.tree import DecisionTreeClassifier
    
    decision_tree_learner = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)

    text_classifier = TextClassifier(estimator=decision_tree_learner, text_cols = ["text1", "text2"], label_cols = ["label"], numeric_cols = ["random_col"], cat_cols = ["bool_col"], extract_word_ngrams=True, extract_char_ngrams=True)
    ```
    
    *** Note that the classifier does not have the ```n_job``` attribute that was present in the logistic regression learner.