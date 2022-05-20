import random
import pdb
from collections import defaultdict

from transformers import AutoTokenizer

from . import constants

def generate_class_prompts(df_sent, n=100):
    df_sent = df_sent.rename(columns={'Reports':'report'})
    df_sent = df_sent[df_sent['report'].map(len)>5].fillna(0).reset_index(drop=True)
    cls_prompts = {}
    for k in constants.CHEXPERT_COMPETITION_TASKS:
        index_a = df_sent[k] == 1
        index_b = (df_sent.drop([k,'report'],axis=1) == 0).all(1)
        sub_sent = df_sent[index_a & index_b]
        if len(sub_sent) > n:
            sub_sent = sub_sent.sample(n)
        cls_prompts[k] = sub_sent['report'].values.tolist()
    return cls_prompts

def generate_chexpert_class_prompts(n = None):
    """Generate text prompts for each CheXpert classification task
    Parameters
    ----------
    n:  int
        number of prompts per class
    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in constants.CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        # randomly sample n prompts for zero-shot classification
        # TODO: we shall make use all the candidate prompts for autoprompt tuning
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts

def process_class_prompts(cls_prompts):
    tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    tokenizer.model_max_length = 77
    cls_prompt_inputs = defaultdict()
    for k,v in cls_prompts.items():
        text_inputs = tokenizer(v, truncation=True, padding=True, return_tensors='pt')
        cls_prompt_inputs[k] = text_inputs
    return cls_prompt_inputs