import random
import pdb
from collections import defaultdict

from transformers import AutoTokenizer

from . import constants

def generate_class_prompts(df_sent, task=None, n=100):
    '''args:
    df_sent: pd.DataFrame with sentence labels, columns=['Reports', 'task1', 'task2',...]
    task: the specified task to build prompts
    n: number of prompts for each task
    '''
    df_sent = df_sent.fillna(0)
    df_sent = df_sent.loc[df_sent['Reports'].map(len)>4].reset_index(drop=True)
    prompts = {}
    all_tasks = df_sent.columns.tolist()[1:]
    if task is not None:
        if isinstance(task, list):
            target_tasks = task
        else:
            target_tasks = [task]
    else:
        target_tasks = all_tasks

    for task in target_tasks:
        other_tasks = [t for t in all_tasks if t != task]
        df_sub_sent = df_sent[(df_sent[task] == 1) & (df_sent[other_tasks] == 0).all(1)]
        if n is not None:
            if len(df_sub_sent) > n: df_sub_sent = df_sub_sent.sample(n)
        prompts[task] = df_sub_sent['Reports'].values.tolist()
    return prompts

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

def generate_covid_class_prompts(n = None):
    prompts = {}
    for k, v in constants.COVID_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        for k0 in v[keys[0]]:
            for k1 in v[keys[1]]:
                for k2 in v[keys[2]]:
                    for k3 in v[keys[3]]:
                        cls_prompts.append(f"{k0} {k1} {k2} {k3}")

        # randomly sample n prompts for zero-shot classification
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts

def generate_rsna_class_prompts(n = None):
    prompts = {}
    for k, v in constants.RSNA_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        for k0 in v[keys[0]]:
            for k1 in v[keys[1]]:
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        # randomly sample n prompts for zero-shot classification
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


def process_class_prompts_for_tuning(cls_prompts, n_context, class_specific_context):
    tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
    tokenizer.model_max_length = 77

    if class_specific_context:
        context = [f'<prompt_{k}_{i}>' for i in range(n_context) for k in cls_prompts]
        num_added_tokens = tokenizer.add_tokens(context)
        assert num_added_tokens == n_context * len(cls_prompts)
    else:
        context = [f'<prompt_{i}>' for i in range(n_context)]
        num_added_tokens = tokenizer.add_tokens(context)
        assert num_added_tokens == n_context

    cls_prompt_inputs = defaultdict()
    for k, v in cls_prompts.items():
        if class_specific_context:
            prefix = ' '.join([f'<prompt_{k}_{i}>' for i in range(n_context)])
        else:
            prefix = ' '.join([f'<prompt_{i}>' for i in range(n_context)])
        context_v = [f'{prefix} {i}' for i in v]
        text_inputs = tokenizer(context_v, truncation=True, padding=True, return_tensors='pt')
        cls_prompt_inputs[k] = text_inputs
    return cls_prompt_inputs

