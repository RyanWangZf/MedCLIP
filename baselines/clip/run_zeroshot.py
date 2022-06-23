import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPModel

from medclip import constants
from medclip.evaluator import Evaluator
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts, \
    generate_rsna_class_prompts
from medclip.prompts import process_class_prompts

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
ensemble = False

# uncomment the following block for experiments
# dataname = 'chexpert-5x200'
# dataname = 'mimic-5x200'
# dataname = 'covid-test'
# dataname = 'covid-2x200-test'
dataname = 'rsna-balanced-test'
# dataname = 'rsna-2x200-test'


class ZeroShotImageDataset(Dataset):
    def __init__(self,
                 datalist=['chexpert-5x200'],
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
        '''support data list in mimic-5x200, chexpert-5x200, rsna-balanced-test, covid-test
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        self.class_names = class_names

        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class ZeroShotImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5):
        # initialize tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass', 'multilabel', 'binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        # process cls prompts into texts indices
        self.prompt_texts_inputs = process_class_prompts(self.cls_prompts)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass', 'binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1, 3, 1, 1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
        }


class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # load two encoders
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                **kwargs):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        # image encoding
        img_embed = self.model.get_image_features(pixel_values)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        # text encoding
        text_embed = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embed, img_embed.t()) * logit_scale
        logits = logits_per_text.T

        outputs = {
            'img_embeds': img_embed, 'text_embeds': text_embed,
            'logits': logits, 'logits_per_text': logits.T, 'loss_value': None
        }

        return outputs


class CLIPClassifier(nn.Module):
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

            medclip_outputs = self.model(**inputs)
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


model = CLIP()
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
        clf = CLIPClassifier(model, ensemble=ensemble)
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
        clf = CLIPClassifier(model, ensemble=ensemble)
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
        clf = CLIPClassifier(model, ensemble=ensemble)
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
