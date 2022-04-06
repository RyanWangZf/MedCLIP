import pdb, os
import math
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image

from transformers import AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES']='0'

from medclip.modeling_medclip import MedClipModel
from medclip.trainer import Trainer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_config = {
    'batch_size': 32,
    'num_epochs': 3,
    'warmup': 0.01, # the first 1% of training steps are used for warm-up
    'lr': 5e-5,
    'weight_decay': 5e-2,
    'eval_batch_size': 128,
    'eval_steps': 100,
    'save_steps': 500,
}

class ImageTextContrastiveDataset(Dataset):
    def __init__(self, imgtransform) -> None:
        super().__init__()
        self.df = pd.read_csv('./local_data/iuxray-meta.csv')
        self.transform = imgtransform
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self.transform(img).unsqueeze(1)
        return img, row.report
            
    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained('phdf33/trialbert-base')
    tokenizer.model_max_length = 77
    inputs = defaultdict(list)
    report_list = []
    for data in batch:
        inputs['pixel_values'].append(data[0])
        report_list.append(data[1])
    text_inputs = tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')

    inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
    inputs['input_ids'] = text_inputs['input_ids']
    inputs['attention_mask'] = text_inputs['attention_mask']
    return inputs

class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, 
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs,
        ):
        outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_loss=True,
                )
        return_res = {
            'loss_value': outputs['loss_value'],
        }
        return return_res

img_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
)
traindata = ImageTextContrastiveDataset(imgtransform=img_transform)
trainloader = DataLoader(traindata, batch_size=train_config['batch_size'], collate_fn=collate_fn)

model = MedClipModel()
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
    use_amp=True,
    )
print('done')








