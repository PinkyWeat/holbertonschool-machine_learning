#!/usr/bin/env python3
""" NLP - Metrics Processing """
import numpy as np
from collections import Counter


def n_grams(sequence, n):
    """Returns the n-grams of a sequence."""
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence """

    ref_lengths = [len(ref) for ref in references]
    sen_length = len(sentence)
    closest_ref_length = min(ref_lengths, key=lambda ref_len:
                             (abs(ref_len - sen_length), ref_len))
    brevity_penalty = np.exp(1 - closest_ref_length / sen_length) \
        if sen_length < closest_ref_length else 1

    precisions = []

    for i in range(1, n + 1):
        sentence_ngrams = Counter(n_grams(sentence, i))
        max_counts = Counter()

        for reference in references:
            reference_ngrams = Counter(n_grams(reference, i))
            for ngram in sentence_ngrams:
                max_counts[ngram] = max(max_counts[ngram],
                                        reference_ngrams[ngram])

        clipped_counts = {ngram: min(count, max_counts[ngram]) for
                          ngram, count in sentence_ngrams.items()}
        precision = (sum(clipped_counts.values()) /
                     sum(sentence_ngrams.values()))
        precisions.append(precision)

    geometric_mean = np.exp(np.mean([np.log(p) for p in precisions if p > 0]))

    BLEU_score = brevity_penalty * geometric_mean
    return BLEU_score
