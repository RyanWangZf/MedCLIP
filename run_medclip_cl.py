import pdb, os

from pyparsing import col
os.environ['CUDA_VISIBLE_DEVICES']='1'
from collections import defaultdict
import requests
import math

from PIL import Image
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from medclip.dataset import IUXRayDataset, IUXRaySentenceDataset
from medclip.dataset import IUXRayImageTextCollator, IUXRayAbnormalNormalCollator, IUXRayFrontalLateralCollator, IUXRayTextCollator
from medclip.modeling_clip import MedCLIPModel
from medclip.losses import ImageTextContrastiveLoss, ImageImageContrastiveLoss
from medclip.trainer import Trainer
from medclip.evaluator import Evaluator

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_config = {
    'batch_size': 64,
    'num_epochs': 100,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 5e-5,
    'weight_decay': 1e-5,
    'eval_batch_size': 128,
    'eval_steps': 100,
    }
model_save_path = f'./checkpoints/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# #########
# define three contrastive loss models
# #########
model = MedCLIPModel()
model = model.to(device)
momentum_model = MedCLIPModel()
momentum_model = momentum_model.to(device)

# #########
# define the evaluator
# #########
model.load_state_dict(torch.load('./checkpoints/pytorch_model.bin'))
eval_data = IUXRayDataset('./data/IU_XRay', 'val')
collate_fn = IUXRayImageTextCollator(img_mean=eval_data.img_mean, img_std=eval_data.img_std, is_train=False)
eval_dataloader = DataLoader(eval_data, batch_size=train_config['eval_batch_size'], shuffle=False, collate_fn=collate_fn)
sentence_data = IUXRaySentenceDataset('./data/IU_XRay')
text_collate_fn = IUXRayTextCollator()
sentence_dataloader = DataLoader(sentence_data, 256, shuffle=False, collate_fn=text_collate_fn)
evaluator = Evaluator(model, eval_dataloader, sentence_dataloader)

# image-text pair CL
training_data = IUXRayDataset('./data/IU_XRay','train')
collate_fn = IUXRayImageTextCollator(img_mean=training_data.img_mean, img_std=training_data.img_std, is_train=True)
dataloader_image_text = DataLoader(training_data, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
train_loss_image_text = ImageTextContrastiveLoss(model)
warmup_steps = math.ceil(len(training_data) * train_config['num_epochs'] * train_config['warmup']) #10% of train data for warm-up

# abnormal-normal pair CL + memory banking (moco V3)
training_data = IUXRayDataset('./data/IU_XRay', 'train')
collate_fn = IUXRayAbnormalNormalCollator(img_mean=training_data.img_mean, img_std=training_data.img_std, is_train=True)
dataloader_abnormal = DataLoader(training_data, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
train_loss_abnormal = ImageImageContrastiveLoss(model, momentum_model)

# frontal-lateral paired CL + memory banks (moco V3)
training_data = IUXRayDataset('./data/IU_XRay', 'train')
collate_fn = IUXRayFrontalLateralCollator(img_mean=training_data.img_mean, img_std=training_data.img_std, is_train=True)
dataloader_frontal = DataLoader(training_data, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
train_loss_frontal = ImageImageContrastiveLoss(model, momentum_model)

train_objectives = [
    (dataloader_image_text, train_loss_image_text),
    (dataloader_abnormal, train_loss_abnormal),
    (dataloader_frontal, train_loss_frontal),
]

# TODO fix checkpoint save and evaluation in trainer
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
    evaluator=evaluator,
    use_amp=True,
    )
print('done')



