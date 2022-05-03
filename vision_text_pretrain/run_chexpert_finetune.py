import pdb, os
import math
import random
os.environ['CUDA_VISIBLE_DEVICES']='1'

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip.modeling_medclip import MedClipModel, MedClipVisionModel, MedClipClassifier
from medclip.dataset import SuperviseImageDataset, SuperviseImageCollator
from medclip import constants
from medclip.losses import ImageSuperviseLoss
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

# define data name
# dataname = 'chexpert-5x200'
# dataname = 'iuxray-5x200'
# dataname = 'mimic-5x200'
# dataname = 'covid19-balance'
# dataname = 'rsna'
# mode = 'binary'

# configuration
# dataname = 'chexpert-finetune'
dataname = 'chexpert-frontal'
mode = 'multilabel'
tasks = constants.CHEXPERT_COMPETITION_TASKS
num_class = 5

train_config = {
    'batch_size': 64,
    'num_epochs': 10,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 1e-5, # 5e-6
    'weight_decay': 0, # 1e-6
    'eval_batch_size': 256,
    'eval_steps': 10,
    'save_steps': 10,
    'max_grad_norm': 0.25,
    'scheduler':'warmupcosine',
}

# for name, param in clf.named_parameters():
#     if name not in ['fc.weight','fc.bias']:
#         param.requires_grad = False

# build dataloader
transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2,0.2),
                transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD])],
            )
train_data = SuperviseImageDataset([f'{dataname}-0.01-train'],
    class_names=tasks,
    imgtransform=transform)

trainloader = DataLoader(train_data, batch_size=train_config['batch_size'], 
    shuffle=True, 
    collate_fn=SuperviseImageCollator(mode=mode),
    num_workers=12,
    )
val_data = SuperviseImageDataset([f'{dataname}-test'],
    class_names=tasks,
    )
valloader = DataLoader(val_data, batch_size=train_config['eval_batch_size'], 
    shuffle=False, 
    collate_fn=SuperviseImageCollator(mode=mode),
    num_workers=8,
    )

# compute pos_weight for BCE loss
df = train_data.df
pos_weight = []
for task in tasks:
    pos_weight_ = (df[task] == 0).sum() / (df[task]==1).sum()
    pos_weight.append(pos_weight_)
pos_weight = torch.tensor(pos_weight)
print(pos_weight)
pos_weight=None # deactivate weight for BCE loss
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# load pretrained model and build classifier
vision_model = MedClipVisionModel(
    medclip_checkpoint='./checkpoints/vision_text_pretrain/25000', # 25000
)
clf = MedClipClassifier(
    vision_model,
    num_class=num_class,
    mode=mode,
    )
loss_model = ImageSuperviseLoss(clf, loss_fn=loss_fn)
loss_model.cuda()

# build objective
train_objectives = [(trainloader, loss_model, 1)]
model_save_path = f'./checkpoints/{dataname}-frontal'

# build trainer
trainer = Trainer()

evaluator = Evaluator(
    medclip_clf=clf,
    eval_dataloader=valloader,
    mode=mode,
)
trainer.train(
    clf,
    train_objectives=train_objectives,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr':train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    evaluator=evaluator,
    scheduler=train_config['scheduler'],
    eval_dataloader=valloader,
    max_grad_norm=train_config['max_grad_norm'],
    use_amp=False,
    )