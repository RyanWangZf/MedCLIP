import os
import sys
from collections import OrderedDict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
# from pyciderevalcap.ciderD.ciderD import CiderD


def evaluate(gts, res):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    agg_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if type(score) == list:
            for m, s in zip(method, score):
                agg_scores[m] = s
        else:
            agg_scores[method] = score

    return agg_scores

