#!/usr/bin/env python3
""" Semantic Search """
import os
import numpy as np
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    """ performs semantic search on a corpus of documents """
    sentence = [sentence]
    docs = []
    model = hub.load(
        "https://www.kaggle.com/models/google/universal-sentence-encoder/" +
        "frameworks/TensorFlow2/variations/large/versions/2")

    for file in os.listdir(corpus_path):
        if not file.endswith('.md'):
            continue
        with open(corpus_path + "/" + file, "r", encoding='utf-8') as f:
            docs.append(f.read())

    doc_embeddings = model(docs)
    sentence_embedding = model(sentence)

    similarity = cosine_similarity(sentence_embedding, doc_embeddings)
    match = np.argmax(similarity)

    return docs[match]
