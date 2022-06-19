import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from medclip import constants
from medclip.dataset import ZeroShotImageCollator
from medclip.dataset import ZeroShotImageDataset
from medclip.evaluator import Evaluator
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts, \
    generate_rsna_class_prompts

# configuration
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# setup config
n_runs = 5
ensemble = True

# uncomment the following block for experiments
# dataname = 'chexpert-5x200'
# dataname = 'mimic-5x200'
# dataname = 'covid-test'
# dataname = 'covid-2x200-test'
# dataname = 'rsna-balanced-test'
dataname = 'rsna-2x200-test'


class ConVIRT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # load two encoders
        self.image_model = models.resnet50(pretrained=True)
        num_fts = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(num_fts, 512)  # projection head

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
            'img_embeds': img_embed, 'text_embeds': text_embed,
            'logits': logits, 'logits_per_text': logits.T, 'loss_value': None
        }
        if return_loss:
            # compute infonce loss
            outputs['loss_value'] = self.compute_loss(logits)
        return outputs

    def compute_loss(self, logits):
        loss_fn = nn.CrossEntropyLoss()
        image_loss = loss_fn(logits, torch.arange(len(logits), device=logits.device))
        caption_loss = loss_fn(logits.T, torch.arange(len(logits.T), device=logits.device))
        loss_value = self.lamb * image_loss + (1 - self.lamb) * caption_loss
        return loss_value


class ConVIRTClassifier(nn.Module):
    def __init__(self, model, ensemble=True):
        super().__init__()
        self.model = model
        self.ensemble = ensemble

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values': pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            medclip_outputs = self.model(**inputs, return_loss=False)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            if self.ensemble:
                cls_sim = torch.mean(logits, 1)  # equivalent to prompt ensembling
            else:
                cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs


model = ConVIRT()
state_dict = torch.load(f'./pretrained/checkpoints/best/{constants.WEIGHTS_NAME}')
model.load_state_dict(state_dict)
model.cuda()

df_sent = pd.read_csv('./local_data/sentence-label.csv', index_col=0)

metrc_list = defaultdict(list)
for i in range(n_runs):
    if dataname in ['chexpert-5x200', 'mimic-5x200']:
        cls_prompts = generate_chexpert_class_prompts(n=10)
        mode = 'multiclass'
        val_data = ZeroShotImageDataset([dataname], class_names=constants.CHEXPERT_COMPETITION_TASKS)
        val_collate_fn = ZeroShotImageCollator(mode=mode, cls_prompts=cls_prompts)
        eval_dataloader = DataLoader(
            val_data,
            batch_size=128,
            collate_fn=val_collate_fn,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        clf = ConVIRTClassifier(model, ensemble=ensemble)
        evaluator = Evaluator(
            medclip_clf=clf,
            eval_dataloader=eval_dataloader,
            mode=mode,
        )

    elif dataname in ['covid-test', 'covid-2x200-test']:
        cls_prompts = generate_class_prompts(df_sent, ['No Finding'], n=10)
        covid_prompts = generate_covid_class_prompts(n=10)
        cls_prompts.update(covid_prompts)
        mode = 'binary'
        val_data = ZeroShotImageDataset([dataname], class_names=constants.COVID_TASKS)
        val_collate_fn = ZeroShotImageCollator(mode=mode, cls_prompts=cls_prompts)
        eval_dataloader = DataLoader(
            val_data,
            batch_size=128,
            collate_fn=val_collate_fn,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        clf = ConVIRTClassifier(model, ensemble=ensemble)
        evaluator = Evaluator(
            medclip_clf=clf,
            eval_dataloader=eval_dataloader,
            mode=mode,
        )

    elif dataname in ['rsna-balanced-test', 'rsna-2x200-test']:
        cls_prompts = generate_class_prompts(df_sent, ['No Finding'], n=10)
        rsna_prompts = generate_rsna_class_prompts(n=10)
        cls_prompts.update(rsna_prompts)
        mode = 'binary'
        val_data = ZeroShotImageDataset([dataname], class_names=constants.RSNA_TASKS)
        val_collate_fn = ZeroShotImageCollator(mode=mode, cls_prompts=cls_prompts)
        eval_dataloader = DataLoader(
            val_data,
            batch_size=128,
            collate_fn=val_collate_fn,
            shuffle=False,
            pin_memory=False,
            num_workers=0,
        )
        clf = ConVIRTClassifier(model, ensemble=ensemble)
        evaluator = Evaluator(
            medclip_clf=clf,
            eval_dataloader=eval_dataloader,
            mode=mode,
        )

    else:
        raise NotImplementedError

    res = evaluator.evaluate()
    for key in res.keys():
        if key not in ['pred', 'labels']:
            print(f'{key}: {res[key]}')
            metrc_list[key].append(res[key])

for key, value in metrc_list.items():
    print('{} mean: {:.4f}, std: {:.2f}'.format(key, np.mean(value), np.std(value)))
