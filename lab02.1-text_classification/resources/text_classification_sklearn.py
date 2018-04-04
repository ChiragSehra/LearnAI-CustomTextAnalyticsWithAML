# coding: utf-8

# Import Packages 
import os
import tatk
import pandas as pd
import numpy as np
from azureml.logging import get_azureml_logger
from tatk.utils import load_newsgroups_data, data_dir, dictionaries_dir, models_dir
from tatk.feature_extraction import NGramsVectorizer
from sklearn.linear_model import LogisticRegression
from tatk.pipelines.text_classification.text_classifier import TextClassifier

logger = get_azureml_logger()

if __name__ == '__main__':

    # Download the benchmark newsgroup training and test data sets.
    X_train, y_train, X_test, y_test = load_newsgroups_data(data_dir)       

    # Copy text column twice, add a numeric column and a boolean with random values  
    random_col = np.random.random_sample(size=(len(X_train),))
    bool_col = [False] *len(X_train)
    df_train = pd.DataFrame({"text1":X_train, "text2":X_train, "bool_col":bool_col, "random_col":random_col, "label":y_train})

    random_col = np.random.random_sample(size=(len(X_test),))
    bool_col = [False] *len(X_test)
    df_test = pd.DataFrame({"text1":X_test, "text2":X_test, "bool_col":bool_col, "random_col":random_col, "label":y_test})

    print("df_train.shape= {}".format(df_train.shape))
    print("df_test.shape= {}".format(df_test.shape))


    # Model training
    # --------------
    # Train a  Scikit-learn text classification model using One-versus-Rest LogisticRegression learning algorithm.

    # Define the logistic regression learner

    log_reg_learner =  LogisticRegression(penalty='l2', dual=False, tol=0.0001, 
                                C=1.0, fit_intercept=True, intercept_scaling=1, 
                                class_weight=None, random_state=None, 
                                solver='lbfgs', max_iter=100, multi_class='ovr',
                                verbose=1, warm_start=False, n_jobs=3) 

    text_classifier = TextClassifier(estimator=log_reg_learner, 
                                    text_cols = ["text1", "text2"], 
                                    label_cols = ["label"], 
                                    numeric_cols = ["random_col"],
                                    cat_cols = ["bool_col"], 
                                    extract_word_ngrams=True, extract_char_ngrams=True)


    # Train the model using the default parameters of the package (word unigrams, bigrams and character 4-grams)
    text_classifier.fit(df_train)  


    # To read the parameters of the different pipeline steps, get step params by step name in the pipeline
    # Read the parameters of the word n-gram extraction module
    char_ngrams_params = text_classifier.get_step_params_by_name("text1_char_ngrams")        
    print("text1_char_ngrams_params:", char_ngrams_params)

    # Test the classifier

    df_test = pd.DataFrame({"text1":X_test, "text2":X_test, "bool_col":bool_col, "random_col":random_col, "label":y_test})
    df_test = text_classifier.predict(df_test)
    print(df_test.head())
    text_classifier.evaluate(df_test)


    working_dir = os.path.join(data_dir, 'outputs')  
    print("os dir:", working_dir)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # you can save the trained model as a folder or a zip file
    model_file = os.path.join(working_dir, 'sk_model.zip')    
    text_classifier.save(model_file)
    # %azureml upload outputs/models/sk_model.zip

    # for debugging, you can save the word n-grams vocabulary to a text file
    word_vocab_file_path = os.path.join(working_dir, 'word_ngrams_vocabulary.tsv')
    text_classifier.get_step_by_name("text1_word_ngrams").save_vocabulary(word_vocab_file_path) 
    # %azureml upload outputs/dictionaries/word_ngrams_vocabulary.pkl

    # for debugging, you can save the character n-grams vocabulary to a text file
    char_vocab_file_path = os.path.join(working_dir, 'char_ngrams_vocabulary.tsv')
    text_classifier.get_step_by_name("text1_char_ngrams").save_vocabulary(char_vocab_file_path) 
    # %azureml upload outputs/dictionaries/char_ngrams_vocabulary.pkl

    loaded_text_classifier = TextClassifier.load(model_file)
    word_ngram_vocab = NGramsVectorizer.load_vocabulary(word_vocab_file_path)
    char_ngram_vocab = NGramsVectorizer.load_vocabulary(char_vocab_file_path)
    loaded_text_classifier.evaluate(df_test)