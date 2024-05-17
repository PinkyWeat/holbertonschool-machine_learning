#!/usr/bin/env python3
""" Word Embeddings """


def gensim_to_keras(model):
    """ converts a gensim word2vec model to a keras Embedding layer """
    return model.wv.get_keras_embedding(True)
