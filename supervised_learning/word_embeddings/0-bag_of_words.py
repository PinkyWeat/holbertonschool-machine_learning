#!/usr/bin/env python3
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ creates a bag of words embedding matrix """

    vectorizeMe = CountVectorizer(vocabulary=vocab)
    embeddings = vectorizeMe.fit_transform(sentences).toarray()
    features = vectorizeMe.get_feature_names_out()

    return embeddings, features
