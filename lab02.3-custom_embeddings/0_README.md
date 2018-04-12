# Custom Embeddings

This hands-on lab demonstrates how you can build custom embeddings with TATK using two state-of-the-art word embedding methods - word2Vec and fastText.

In this lab, we will:
- Understand word embeddings and the technique(s) behind getting word representation
- Train word2Vec/fastText models and evaluate
- Save and load pipeline for additional training
- Identify semantically similar terms

### Learning Objectives ###

The objectives of this lab are to:

- Understand the concept of word representations for predicting surrounding context
- Understand the difference between the two embedding methods - word2Vec and fastText
- Learn how to create word embedding pipeline, train and evaluate the models
- Understand how you can identify semantically similar terms using word embeddings

## Word Embeddings

Word Embedding turns text into a vector space. This transformation is necessary because many machine learning algorithms require their input to be vectors of continuous values. Word Embedding is used to map words or phrases from a vocabulary to a corresponding vector of real numbers. The vectors from Word Embedding preserve these similarities, so words that regularly occur nearby in text will also be in close proximity in vector space.

### Word2Vec

Word2Vec is a neural embedding model and learns vectors only for complete words found in the training corpus. Word2Vec is very much like [GloVe](https://nlp.stanford.edu/projects/glove/)  - both treat words as the smallest unit to train on. Word2Vec tries to capture co-occurrence one window at a time whereas GloVe captures counts of overall statistics.

### FastText

FastText learns vectors for the n-grams that are found within each word, as well as each complete word. At each training step in FastText, the mean of the target word vector and its component n-gram vectors are used for training. 

For example, the word vector "apple" is a sum of the vectors of the n-grams “<ap”, “app”, ”appl”, ”apple”, ”apple>”, “ppl”, “pple”, ”pple>”, “ple”, ”ple>”, ”le>” (assuming hyperparameters for smallest ngram[minn] is 3 and largest ngram[maxn] is 6). 

### Execution

1. Launch Azure Machine Learning Workbench and open CLI by selecting File -> Open Command Prompt.

2. The two scripts for building word embeddings are:
    
    a. ```resources/hotels_Word2Vec.py``` (uses Word2Vec for embeddings)
    
    b. ```resources/hotels_fastText.py``` (uses fasttext for embeddings)

Execute the Word2Vec script by runing the below command and walk through the code:

```az ml experiment submit -c local hotels_Word2Vec.py```

## Dataset to create Embeddings

The dataset used in this lab to illustrate the concept of word embeddings captures manually curated hotel descriptions. In this lab, we will explore how we can get semantically similar terms in the hotel domain. Example rows of the dataset are of the below format:

| Id | Description |
| ----- | ----- |
| 1 | This beautiful villa features a spacious dining area and a patio facing the Santa Monica beach. Just minutes away from the beach, feel the fresh breeze and enjoy being around a vibrant shopping center.|
| 2 | Located in the hills with spectacular sea views, this hotel is ideal for an outdoor experience. If you enjoy walking, running or hiking, this is an ideal location |
| 3 | Located on the strip and minutes away from the heart of the entertainment center, this hotel gives you the ultimate Vegas experience. |


In the two python scripts, the below snippet points to the dataset. Ensure that ````C:\tatk\resources```` is created (if not already) and place _hotel_data.csv_ in the folder.

````python
    local_path = r'C:\tatk\resources'
    file_name = "hotelDataSet\hotel_data.csv"
````

## Word2Vec model pipeline

The word2Vec model pipeline can be easily created using Word2VecModel. We can initialize the pipeline using default parameters. In addition, ````detect_sentences = True```` allows for sentence segmentation. For more advanced cleaning tasks, you can also pass regular expressions as an argument to Word2VecModel.

````python
    Word2Vec_model = Word2VecModel(input_col = 'Description', regex = None, detect_sentences = True)
````

In addition, we will also use the skip gram model by setting ````use_skipgram = 1```` in the below line.

````python
    Word2Vec_model.set_step_params_by_name('vectorizer', use_skipgram = 1) 
````

Firstly, a skip gram model identifies every word (i.e. a focus word) in a large corpora. For each focus word, it takes one-by-one the words that surround it within a defined "window". These words are then fed into a neural network to predict the probability for each word to appear in the window around the focus word.

## Model and embeddings file

The word2Vec model and embeddings file can be saved as shown below.

````python
    pipeline_path = os.path.join(models_dir, 'Word2Vec_model')
    Word2Vec_model.save(pipeline_path, create_folders_on_path=True)
    Word2Vec_model2 = Word2VecModel.load(pipeline_path)

    embeddings_file_path = os.path.join(models_dir, 'Word2Vec_embeddings.txt')
    Word2Vec_model2.save_embeddings(embeddings_file_path)
````

Note that the embeddings file is in textual format. Open the embeddings file using a text editor to view the content. A sample of embeddings file is as follows where you have term followed by the vector.

````hotel 0.113766 -0.261705 -0.321098 ...````

## OOV

It is very common to come accross words that did not occur in the training corpus. This is also known as Out of Vocabulary (OOV) words. The toolkit provides flexibility to train a model that includes an explicit symbol for an unknown word (\<UNK\>). More specifically, in the below line, we can load embeddings to memory with ````include_unk = True```` to add OOV treatment.

````python
    vectorizer = Word2VecVectorizer.load_embeddings(embeddings_file_path, include_unk = True, unk_method = 'rnd', unk_vector = None, unk_word = '<UNK>')
````

For unk_method, we can have:
- a random vector ('rnd')
- mean of all the vocabulary ('mean')
- vector of all zeros ('zero')

For unk_vector, you can chose to provide your own vector for unknown words.
                
For unk_word, it is the string representation for the unknown words.



## Embedding Lookup

The toolkit provides an easy way to perform embedding lookup using _most_similar_ function. For example, to look up terms that are semantically similar to the word 'beach', you can call the below line. 

````python
    vectorizer.embedding_table.most_similar('beach')
````

This will produce terms that are similar along with a score as shown below:

````
[('sandy', 0.777091383934021), ('beaches', 0.716056227684021), ('sand', 0.6506246328353882), ('rocky', 0.650464653968811), ('pebble', 0.6420736312866211), ('sands', 0.6214091181755066), ('dunes', 0.6139158606529236), ('bay', 0.6124751567840576), ('waters', 0.6093554496765137), ('seafront', 0.6078104972839355)]
````

Note that many of the terms such as 'sandy', 'beaches', 'pebble', 'waters', 'seafront' are semantically very similar to beach.

## Exercise (Optional)

1. Run the fastText script (hotels_fastText.py) now and observe the semantically similar terms. Are they different from what was obtained using word2Vec?

2. Can you evaluate the quality of the semantic similarity on benchmark datasets such as [WordSimilarity-353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/). This dataset consists of English word pairs along with human-assigned similarity/relatedness judgements. You can use a measure such as Pearson's correlation to measure the quality of the embeddings obtained.

An easy way of getting similarity score between two terms (for example, "beach" and "sand") is as follows:

````python
vectorizer.embedding_table.similarity("beach", "sand"))
````

3. The approach used in this lab works well for word-word similarity. How can we extend this to word-sentence similarity? 