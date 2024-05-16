#!/usr/bin/env python3
""" Word Embeddings """
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix"""

    if vocab is None:
        vocab = []
        for sentence in sentences:
            # Lowercase, replace non-alphanumeric characters with space, remove single characters
            cleaned_sentence = re.sub(r"\b\w{1}\b", "", re.sub(r"[^a-zA-Z0-9\s]", " ", sentence.lower()))
            vocab.extend(cleaned_sentence.split())
        vocab = sorted(list(set(vocab)))  # Remove duplicates and sort

    embeddings = np.zeros((len(sentences), len(vocab)))  # Initialize the embedding matrix

    for i, sentence in enumerate(sentences):
        # Apply the same cleaning process for each word in the sentence
        words = re.sub(r"\b\w{1}\b", "", re.sub(r"[^a-zA-Z0-9\s]", " ", sentence.lower())).split()
        for word in words:
            if word in vocab:
                embeddings[i][vocab.index(word)] += 1  # Increment the count at the corresponding index

    return embeddings.astype(int), vocab
