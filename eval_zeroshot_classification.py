from collections import defaultdict
from email.policy import default
import pdb, os
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from medclip.modeling_medclip import MedClipModel, MedClipPromptClassifier
from medclip.dataset import ImageTextContrastiveDataset, ZeroShotImageDataset
from medclip.dataset import ImageTextContrastiveCollator, ZeroShotImageCollator
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts
from medclip import constants

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set device
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# build medclip model and load from the pretrained checkpoint
model = MedClipModel(
    checkpoint='./checkpoints/vision_text_pretrain/25000',
    )
model.cuda()

# generate class prompts by sampling from sentences
df_sent = pd.read_csv('./local_data/sentence-label.csv', index_col=0)

dataname = 'chexpert-5x200'
# dataname = 'iuxray-5x200'
# dataname = 'mimic-5x200'
tasks = constants.CHEXPERT_COMPETITION_TASKS

# dataname = 'covid19-balance-test'
# tasks = constants.COVID_TASKS

# build evaluator by passing the task and dataname
val_data = ZeroShotImageDataset([dataname], class_names=tasks)

# do evaluation for multiple runs
metrc_list = defaultdict(list)
for i in range(5):
    # ##########
    # for covid19 data
    # cls_prompts = generate_class_prompts(df_sent, ['No Finding'], n=10)
    # covid_prompts = generate_covid_class_prompts(n=10)
    # cls_prompts.update(covid_prompts)
    # val_collate_fn = ZeroShotImageCollator(mode='binary', cls_prompts=cls_prompts)
    # eval_dataloader = DataLoader(val_data,
    #     batch_size=128,
    #     collate_fn=val_collate_fn,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=8,
    #     )
    # medclip_clf = MedClipPromptClassifier(model, ensemble=False)
    # evaluator = Evaluator(
    #     medclip_clf=medclip_clf,
    #     eval_dataloader=eval_dataloader,
    #     mode='binary',
    # )
    # ##########

    # ##########
    # for iuxray and chexpert and mimic
    cls_prompts = generate_chexpert_class_prompts(n=10)
    cls_prompts = generate_class_prompts(df_sent, task=constants.CHEXPERT_COMPETITION_TASKS, n=10)

    val_collate_fn = ZeroShotImageCollator(cls_prompts=cls_prompts, mode='multiclass')
    eval_dataloader = DataLoader(val_data,
        batch_size=128,
        collate_fn=val_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        )
    medclip_clf = MedClipPromptClassifier(model, ensemble=True)
    evaluator = Evaluator(
        medclip_clf=medclip_clf,
        eval_dataloader=eval_dataloader,
        mode='multiclass',
    )
    # ##########

    res = evaluator.evaluate()
    for key in res.keys():
        if key not in ['pred','labels']:
            print(f'{key}: {res[key]}')
            metrc_list[key].append(res[key])

for key,value in metrc_list.items():
    print('{} mean: {:.4f}, std: {:.2f}'.format(key, np.mean(value), np.std(value)))
