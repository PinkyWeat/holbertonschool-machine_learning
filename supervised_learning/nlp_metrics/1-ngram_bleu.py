#!/usr/bin/env python3
""" NLP - Metrics Processing """
import numpy as np


def ngram_bleu(references, sentence, n):
    """ calculates the n-gram BLEU score for a sentence """
    brev_penalty = min(1, np.exp(1 - len(min(references, key=len))
                                 / len(sentence)))
    n_grams = []

    for reference in references:
        n_grams_ref = []
        for i in range(len(sentence) - (n - 1)):
            if (any(sentence[i:i + n] == reference[j:j + n]
                    for j in range(len(reference) -
                                   (n - 1))) and sentence[i:i + n]
                    not in n_grams_ref):
                n_grams_ref.append(sentence[i:i + n])
        n_grams.append(len(n_grams_ref))

    precision = max(n_grams) / (i + 1)

    return brev_penalty * np.exp(np.log(precision))
