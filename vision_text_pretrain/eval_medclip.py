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

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# build medclip model
model = MedClipModel(
    checkpoint='./checkpoints/vision_text_pretrain/45000',
    )
model.cuda()

# build evaluator
val_data = ZeroShotImageDataset(['chexpert-5x200'])

for i in range(5):
    val_collate_fn = ZeroShotImageCollator(n_prompt=10)
    eval_dataloader = DataLoader(val_data,
        batch_size=128,
        collate_fn=val_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        )
    medclip_clf = MedClipPromptClassifier(model)
    evaluator = Evaluator(
        medclip_clf=medclip_clf,
        eval_dataloader=eval_dataloader,
    )
    print(evaluator.evaluate())
