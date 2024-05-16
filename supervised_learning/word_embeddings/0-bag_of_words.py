import numpy as np

def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix"""

    if vocab is None:
        features = set(word for sentence in sentences for word in sentence.lower().split())
        features = sorted(features)
    else:
        features = sorted(vocab)

    features_dict = {word: idx for idx, word in enumerate(features)}

    embeddings = np.zeros((len(sentences), len(features)))

    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        for word in words:
            if word in features_dict:
                embeddings[i, features_dict[word]] += 1

    return embeddings, features
