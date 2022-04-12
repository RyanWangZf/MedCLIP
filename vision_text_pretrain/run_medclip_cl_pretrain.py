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

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# set training configurations
train_config = {
    'batch_size': 128,
    'num_epochs': 10,
    'warmup': 0.01, # the first 1% of training steps are used for warm-up
    'lr': 5e-4,
    'weight_decay': 1e-2,
    'eval_batch_size': 128,
    'eval_steps': 100,
    'save_steps': 500,
}


# only pretrain on chexpert train data and mimic-cxr data
# do zero-shot training on chexpert-5x200 and iuxray
datalist = [
    'chexpert-train',
    'mimic-cxr',
]
traindata = ImageTextContrastiveDataset(datalist=datalist)
trainloader = DataLoader(traindata, 
    batch_size=train_config['batch_size'], 
    collate_fn=traindata.collate_fn, 
    shuffle=True,
    pin_memory=False,
    num_workers=0,
    )


# build medclip model
# model = MedClipModel(
#     vision_checkpoint='./checkpoints/vision_pretrain'
#     )
model = MedClipModel()
model.cuda()

val_data = ZeroShotImageDataset(['chexpert-5x200'])
eval_dataloader = DataLoader(val_data,
    batch_size=train_config['eval_batch_size'],
    collate_fn=val_data.collate_fn,
    shuffle=False,
    pin_memory=False,
    num_workers=0,
    )
medclip_clf = MedClipPromptClassifier(model)

# build evaluator
evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader,
)

# build loss models
loss_model = ImageTextContrastiveLoss(model)
loss_model.cuda()
train_objectives = [
    (trainloader, loss_model, 1),
]
warmup_steps = math.ceil(len(traindata) * train_config['num_epochs'] * train_config['warmup']) #10% of train data for warm-up
model_save_path = f'./checkpoints/vision_text_pretrain'
trainer = Trainer()
trainer.train(
    model,
    train_objectives=train_objectives,
    warmup_steps=warmup_steps,
    epochs=train_config['num_epochs'],
    optimizer_params={'lr':train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    evaluator=evaluator,
    eval_dataloader=eval_dataloader,
    use_amp=True,
    )
print('done')








