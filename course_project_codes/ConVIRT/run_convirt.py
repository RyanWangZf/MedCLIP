from cmath import log
import pdb, os
import math
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
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


# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
os.environ['CUDA_VISIBLE_DEVICES']='1'
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class ImageTextContrastiveDataset(Dataset):
    def __init__(self, datalist=['chexpert', 'mimic-cxr', 'iuxray'], imgtransform=None) -> None:
        super().__init__()
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)
        self.df = self.df[self.df['report'].map(len)>0] # remove no report data
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)), # convirt take 224x224 inputs
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self.transform(img).unsqueeze(1)
        report = row.report # original sentences list
        # randomly sample one sentence
        sent_ix = random.randint(0, len(report)-1)
        report = report[sent_ix]
        return img, report

    def __len__(self):
        return len(self.df)

    def create_sent_segments(self, df):
        '''do preprocessing to split raw reports into sentence segments for
        sentence-image contrastive pretraining.
        '''
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            splitter = re.compile("[0-9]+\.")
            report = splitter.split(report)
            reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue
                
                sent = sent.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(sent.lower())
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))
            return study_sent

class ImageTextContrastiveCollator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0])
            report_list.append(data[1])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0).repeat((1,3,1,1))
        # repeat three chanels
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        return inputs

class ConVIRT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # load two encoders
        self.image_model = models.resnet50(pretrained=True)
        num_fts = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(num_fts, 512) # projection head
        
        self.text_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.text_projection_head = nn.Linear(768, 512)

        # hyperparameter
        self.lamb = 0.75
        self.temperature = 0.1

    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        return_loss=True,
        **kwargs):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        # image encoding
        img_embed = self.image_model(pixel_values)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)


        # text encoding
        text_embed = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = text_embed['pooler_output']
        text_embed = self.text_projection_head(text_embed)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        logits = torch.matmul(img_embed, text_embed.t()) / self.temperature

        outputs = {
            'img_embeds':img_embed, 'text_embeds':text_embed,
            'logits':logits, 'logits_per_text':logits.T, 'loss_value':None
        }
        if return_loss:
            # compute infonce loss
            outputs['loss_value'] = self.compute_loss(logits)
        return outputs
        
    def compute_loss(self, logits):
        loss_fn = nn.CrossEntropyLoss()
        image_loss = loss_fn(logits, torch.arange(len(logits), device=logits.device))
        caption_loss = loss_fn(logits.T, torch.arange(len(logits.T), device=logits.device))
        loss_value = self.lamb * image_loss + (1-self.lamb) * caption_loss
        return loss_value

class ConVIRTClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            medclip_outputs = self.model(**inputs, return_loss=False)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            cls_sim = torch.mean(logits, 1) # equivalent to prompt ensembling
            # cls_sim = torch.max(logits, 1)
            class_similarities.append(cls_sim)
            class_names.append(cls_name)
        
        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

# set training configurations
train_config = {
    'batch_size': 100,
    'num_epochs': 10,
    'warmup': 0.01, # the first 1% of training steps are used for warm-up
    'lr': 1e-4,
    'weight_decay': 1e-6,
    'eval_batch_size': 128,
    'eval_steps': 1000,
    'save_steps': 1000,
}

# only pretrain on chexpert train data and mimic-cxr data
# do zero-shot training on chexpert-5x200 and iuxray
datalist = [
    'mimic-cxr',
]
traindata = ImageTextContrastiveDataset(datalist=datalist)
train_collate_fn = ImageTextContrastiveCollator()
trainloader = DataLoader(traindata, 
    batch_size=train_config['batch_size'], 
    collate_fn=train_collate_fn, 
    shuffle=True,
    pin_memory=True,
    num_workers=12,
    )

model = ConVIRT()
model.cuda()

# build evaluator
val_data = ZeroShotImageDataset(['chexpert-5x200'])
val_collate_fn = ZeroShotImageCollator()
eval_dataloader = DataLoader(val_data,
    batch_size=train_config['eval_batch_size'],
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    )
medclip_clf = ConVIRTClassifier(model)
evaluator = Evaluator(
    medclip_clf=medclip_clf,
    eval_dataloader=eval_dataloader,
)

train_objectives = [
    (trainloader, model, 1),
]

model_save_path = f'./checkpoints/covirt_pretrain'
trainer = Trainer()
trainer.train(
    model,
    train_objectives=train_objectives,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr':train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    use_amp=True,
    evaluator=evaluator,
    eval_dataloader=eval_dataloader,
    )
print('done')