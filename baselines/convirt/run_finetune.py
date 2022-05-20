import pdb, os
import math
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer, AutoModel

from medclip.trainer import Trainer
from medclip.evaluator import Evaluator
from medclip.dataset import ZeroShotImageDataset
from medclip.dataset import ZeroShotImageCollator
from medclip import constants

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ImageDataset(Dataset):
    def __init__(self, 
        datalist=['chexpert', 'mimic-cxr', 'iuxray'], 
        imgtransform=None,
        class_names=None,
        ) -> None:
        super().__init__()
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)), # convirt take 224x224 inputs
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else: 
            self.transform = imgtransform
        self.class_names = class_names
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def __len__(self):
        return len(self.df)

class ImageCollator:
    def __init__(self, mode):
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])
        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0).repeat((1,3,1,1))
        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)
        return inputs

class ConVIRTClassifier(nn.Module):
    def __init__(self,
        vision_model,
        num_class=None,
        mode=None,
        ):
        super().__init__()
        self.num_class = num_class
        self.mode = mode.lower()
        input_dim = vision_model.fc.in_features
        if num_class > 2:
            if mode == 'multiclass':
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
            vision_model.fc = nn.Linear(input_dim, num_class)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            vision_model.fc = nn.Linear(input_dim, 1)
        self.model = vision_model
    
    def forward(self, 
        pixel_values, 
        labels=None,
        return_loss=True,
        **kwargs,
        ):
        outputs = defaultdict()
        pixel_values = pixel_values.cuda()
        # take embeddings before the projection head
        logits = self.model(pixel_values)
        outputs['logits'] = logits
        if labels is not None and return_loss:
            labels = labels.cuda().float()
            if len(labels.shape) == 1: labels = labels.view(-1,1)
            if self.mode == 'multiclass': labels = labels.flatten().long()
            loss = self.loss_fn(logits, labels)
            outputs['loss_value'] = loss
        return outputs


# configuration
# dataname = 'chexpert-5x200'
# dataname = 'mimic-5x200'
dataname = 'iuxray-5x200'
mode = 'multiclass'
tasks = constants.CHEXPERT_COMPETITION_TASKS
num_class = 5

# dataname = 'covid19-balance'
# mode = 'binary'
# tasks = constants.COVID_TASKS
# num_class = 2

# dataname = 'rsna'
# mode = 'binary'
# tasks = constants.RSNA_TASKS
# num_class = 2

# dataname = 'iuxray'
# mode = 'multilabel'
# tasks = constants.CHEXPERT_COMPETITION_TASKS
# num_class = 5

train_config = {
    'batch_size': 64,
    'num_epochs': 20,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 5e-5,
    'weight_decay': 0,
    'eval_batch_size': 256,
    'eval_steps': 10,
    'save_steps': 10,
}

vision_model  = models.resnet50()
new_state_dict = {}
state_dict = torch.load(f'./pretrained/{constants.WEIGHTS_NAME}')
for key in state_dict.keys():
    if 'image_model' in key and 'fc' not in key:
        new_state_dict[key.replace('image_model.','')] = state_dict[key]
missing_keys, unexpected_keys = vision_model.load_state_dict(new_state_dict, strict=False)
clf = ConVIRTClassifier(vision_model=vision_model,num_class=num_class, mode=mode)
clf.cuda()

train_data = ImageDataset([f'{dataname}-train'],
    class_names=tasks)
trainloader = DataLoader(train_data, batch_size=train_config['batch_size'], 
    shuffle=True, 
    collate_fn=ImageCollator(mode=mode),
    num_workers=8,
    )
val_data = ImageDataset([f'{dataname}-test'],
    class_names=tasks,
    )
valloader = DataLoader(val_data, batch_size=train_config['eval_batch_size'], 
    shuffle=False, 
    collate_fn=ImageCollator(mode=mode),
    num_workers=4,
    )

# build objective
train_objectives = [(trainloader, clf, 1)]
model_save_path = f'./checkpoints/{dataname}-finetune'

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
    eval_dataloader=valloader,
    use_amp=False,
    )
res = evaluator.evaluate()
for key in res.keys():
    if key not in ['pred','labels']:
        print(f'{key}: {res[key]}')

