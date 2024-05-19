#!/usr/bin/env python3
""" NLP - Metrics Processing """
import numpy as np


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence """
    ref_lens = [len(ref) for ref in references]
    sent_len = len(sentence)
    closest_ref_len = min(ref_lens,
                          key=lambda ref_len: (abs(ref_len - sent_len),
                                               ref_len))

    brevity_penalty = 1 if sent_len > closest_ref_len \
        else np.exp(1 - closest_ref_len / sent_len)

    # maximum precision over all references
    precision = max(sum(word in reference for word in sentence) /
                    sent_len for reference in references)

    return brevity_penalty * precision  # actual BLEU score
