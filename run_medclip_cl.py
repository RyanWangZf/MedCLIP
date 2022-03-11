import pdb, os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from collections import defaultdict
import requests
import math

from PIL import Image
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

from medclip.dataset import IUXRayDataset, IUXRayImageTextCollator, IUXRayAbnormalNormalCollator
from medclip.modeling_clip import MedCLIPModel
from medclip.losses import ImageTextContrastiveLoss
from medclip.trainer import Trainer

train_config = {
    'train_batch_size': 16, # 16 * 5 = 80
    'num_epochs': 1,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 1e-5,
    'weight_decay': 1e-4,
    }
model_save_path = f'./checkpoints/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# #########
# define three contrastive loss models?
# #########
model = MedCLIPModel()

# image-text pair CL
training_data = IUXRayDataset('./data/IU_XRay')
collate_fn = IUXRayImageTextCollator(img_mean=training_data.img_mean, img_std=training_data.img_std, is_train=True)
dataloader = DataLoader(training_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
train_loss_image_text = ImageTextContrastiveLoss(model)
warmup_steps = math.ceil(len(training_data) * train_config['num_epochs'] * train_config['warmup']) #10% of train data for warm-up

# three loss models
# abnormal-normal pair CL + memory banking
# training_data = IUXRayDataset('./data/IU_XRay')
# collate_fn = IUXRayAbnormalNormalCollator(img_mean=training_data.img_mean, img_std=training_data.img_std, is_train=True)
# dataloader = DataLoader(training_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

# test data collator
# for batch in dataloader:
#     pass

# frontal-lateral pair CL
train_objectives = [
    (dataloader, train_loss_image_text),
]

trainer = Trainer()
trainer.train(
    model,
    train_objectives=train_objectives,
    warmup_steps=warmup_steps,
    epochs=train_config['num_epochs'],
    optimizer_params={'lr':train_config['lr']},
    batch_size=train_config['train_batch_size'],
    output_path=model_save_path,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=1000,
    weight_decay=train_config['weight_decay'],
    use_amp=True,
    )
print('done')



