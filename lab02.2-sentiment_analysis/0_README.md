# Sentiment Analysis using CNN and pre-trained word2vec model

This hands-on lab demonstrates the use of CNNs together with a pre-trained embedding model using TATK.

In this lab, we will:
- Utilize pre-trained embedding model that includes word vectors trained on google news for predicting sentiment of movie reviews
- Explore the IMDB dataset by generating summary statistics
- Build a CNN model using TATK's KerasTextClassifier and evaluate the performance of the model
- Save the pipeline, load it and test on unseen reviews to predict sentiment

### Learning Objectives ###

The objectives of this lab are to:

- Understand how you can integrate a pre-trained embedding model in the pipeple
- Understand how to build a CNN model with the pre-trained model for setiment prediction
- Understand how you can use the model for scoring sentiment on unseen reviews