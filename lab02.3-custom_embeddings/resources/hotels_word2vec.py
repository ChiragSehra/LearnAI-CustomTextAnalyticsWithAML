from azureml.logging import get_azureml_logger
from tatk.utils import load_biomedical_data, download_embedding_model, data_dir, dictionaries_dir, models_dir
from tatk.connectors.blob_storage_data_connector import AzureBlobStorageDataConnector
from tatk.pipelines.feature_extraction.word2vec_model import Word2VecModel
from tatk.feature_extraction.word2vec_vectorizer import Word2VecVectorizer
import os
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    logger = get_azureml_logger()


    # Prepare data for modeling and evaluation
    local_path = r'C:\tatk\resources'
    file_name = "hotelDataSet\hotel_data.csv"

    file_path = os.path.join(local_path, file_name)
    data = pd.read_csv(file_path, sep = "~", usecols = ['Id', 'Description'], encoding = "ISO-8859-1").dropna()#read in-memory
    df = data

    # Create the Word2Vec model pipeline
    # Initialize the pipeline with default parameters. No regular expression cleaning is performed, and sentences are detected. 

    word2vec_model = Word2VecModel(input_col = 'Description', regex = None, detect_sentences = True)


    # To get pipeline parameters
    # word2vec_model.get_step_params_by_name('vectorizer')

    word2vec_model.set_step_params_by_name('vectorizer', use_skipgram = 1) 
    word2vec_model.fit(df)

    pipeline_path = os.path.join(models_dir, 'word2vec_model')
    word2vec_model.save(pipeline_path, create_folders_on_path=True)
    word2vec_model2 = Word2VecModel.load(pipeline_path)

    # Saved embeddings file is in textual format and is readable if opened with a text editor
    embeddings_file_path = os.path.join(models_dir, 'word2vec_embeddings.txt')
    word2vec_model2.save_embeddings(embeddings_file_path)


    # Load the embeddings to memory with include_unk set to True to add OOV treatment

    vectorizer = Word2VecVectorizer.load_embeddings(embeddings_file_path, include_unk = True,
                                                    unk_method = 'rnd', unk_vector = None, unk_word = '<UNK>')

    # Embedding Lookup: Get most similar word to a given word.

    print("*** Hotel Terms *** ")
    print(vectorizer.embedding_table.most_similar('hotel'))

    print("*** Beach Terms *** ")
    print(vectorizer.embedding_table.most_similar('beach'))