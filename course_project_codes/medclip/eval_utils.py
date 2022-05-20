import os
import sys
from collections import OrderedDict

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def nlp_evaluate(gts, res):
    '''inputs
    gts: a dict of id:report, groundtruth
    res: a dict of generated reports, id:report, prediction
    '''
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
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