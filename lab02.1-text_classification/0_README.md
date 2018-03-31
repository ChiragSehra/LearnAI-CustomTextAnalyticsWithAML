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

### Getting Started

1. Launch Azure Machine Learning Workbench and open CLI by selecting File -> Open Command Prompt.

2. 

### Datasets

In this lab, we will use two datasets (located at https://aztatksa.blob.core.windows.net/textprivatepreview) for performing text classification.

#### Newsgroups Dataset

The 20 newsgroups collection has become a popular data set for experiments in text mining, such as text classification and text clustering. The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. In this lab, we will be using 4 categories - comp.graphics, rec.autos, sci.med and misc.forsale. You can also opt to load the newsgroup dataset yourself.

#### Sentiment Evaluation Dataset

   | id | text | label |
   | 1 | @killa_1983 - If you ain't doing nothing Satur... |positive |
   | 2 | Pop bottles , make love , thug passion , RED... | positive |
   | 3 | @TheScript_Danny @thescript - St Patricks Day ... | positive |

### Jobs

“log_reg_learner has njobs=3” means that the logistic regression algo will run on 3 threads.
“number of jobs for the pipeline: 6” means that the pipeline execution can run on maximum 6 parallel threads if possible. It depends on the pipeline nodes. It may not have this maximum number of nodes that could be run in parallel.

There is 8 nodes in the pipeline but not all run in parallel. 
Node 1 and Node 2 depend on Node 0. Node 4 and Node 5 depend on Node 3. Node 7 depends on Node 6. 

0 text1_nltk_preprocessor
1 text1_word_ngrams
2 text1_char_ngrams
3 text2_nltk_preprocessor
4 text2_word_ngrams
5 text2_char_ngrams
6 assembler
7 learner
