'''do contrastive pretraining on image encoder only
'''
import pdb, os
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES']='1'

from medclip.modeling_medclip import MedClipVisionModel
from medclip.trainer import Trainer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_config = {
    'batch_size': 60,
    'num_epochs': 20,
    'warmup': 0.01, # the first 1% of training steps are used for warm-up
    'lr': 1e-3,
    'weight_decay': 5e-2,
    'eval_batch_size': 128,
    'eval_steps': 100,
    'save_steps': 500,
}

# pretrain image encoder with pure images
class ImageContrastiveDataset(Dataset):
    def __init__(self, datalist=['mimic-cxr','iuxray','chexpert'], transform1=None,transform2=None) -> None:
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        self.df = df

        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        row  = self.df.iloc[index]
        img = Image.open(row.imgpath)
        x1 = self.transform1(img)
        x2 = self.transform2(img)
        return {'img1':x1, 'img2':x2}

    def __len__(self,):
        return len(self.df)

class VisionModelContrastiveLoss(nn.Module):
    def __init__(self, model, momentum_model) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.model = model
        self.momentum_model = momentum_model
        for param_b, param_m in zip(self.model.parameters(), self.momentum_model.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
            param_m.to(param_b.device)
        self.contrast_temperature=0.2

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.model.parameters(), self.momentum_model.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, img1, img2):
        img1 = img1.cuda()
        img2 = img2.cuda()
        feat1 = self.model(img1)
        feat2 = self.model(img2)

        with torch.no_grad():
            self._update_momentum_encoder(0.99)
            feat1_ng = self.momentum_model(img1)
            feat2_ng = self.momentum_model(img2)
        
        loss = self.contrastive_loss(feat1, feat2_ng) + self.contrastive_loss(feat2,feat1_ng)
        return {'loss_value':loss}

    def contrastive_loss(self, q, k):
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.contrast_temperature
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
    

# define training configuration
img_transform1 = transforms.Compose([
    transforms.Resize((280,280)),
    transforms.RandomCrop(256,256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])
]
)
img_transform2 = transforms.Compose([
    transforms.Resize((280,280)),
    transforms.RandomCrop(256,256),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.1,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])
]

)
traindata = ImageContrastiveDataset(transform1=img_transform1, transform2=img_transform2)
dataloader = DataLoader(traindata, batch_size=train_config['batch_size'], 
    num_workers=12,
    shuffle=True, 
    pin_memory=True)
model = MedClipVisionModel()
model_m = MedClipVisionModel()
loss_model = VisionModelContrastiveLoss(model, model_m)
loss_model.cuda()

train_objectives = [
    (dataloader, loss_model, 1),
]
warmup_steps = math.ceil(len(traindata) * train_config['num_epochs'] * train_config['warmup']) #10% of train data for warm-up
model_save_path = f'./checkpoints/vision_pretrain'
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
    use_amp=True,
    )
print('done')


