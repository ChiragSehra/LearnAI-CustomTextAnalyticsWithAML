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

## Dataset and Models

The dataset is obtained using _load_biomedical_data_ where each sentence is tokenized and annotated. For example, the below annotated sentence

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