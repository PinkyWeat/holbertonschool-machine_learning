#!/usr/bin/env python3
""" Word Embeddings """
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ creates a TF-IDF embedding """
    the_vector = TfidfVectorizer(vocabulary=vocab)
    embed = the_vector.fit_transform(sentences)
    features = the_vector.get_feature_names()
    return embed.toarray(), features
