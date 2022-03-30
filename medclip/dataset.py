import os
from collections import defaultdict
import pdb
import itertools

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor, CLIPTokenizer

import pandas as pd
import numpy as np
from .feature_extractor import MedCLIPFeatureExtractor

class IUXRayDataset(Dataset):
    '''
    # how to crop raw images into patches
    res=rearrange(x_frontal[:,None,:9*224,:11*224], 'b c (h p1) (w p2) -> b (h w c) p1 p2', p1=224,p2=224)
    '''
    _report_sections_ = ['findings','impression','MeSH']
    channel_num = 1 # XRay is a gray scale image
    img_mean = [0.5862785803043838]
    img_std = [0.27950088968644304]
    def __init__(self, datadir, mode=None):
        
        assert mode in ['train','test','val']
        self.image_dir = os.path.join(datadir, './images/images_normalized')
        projection = pd.read_csv(os.path.join(datadir, 'indiana_projections.csv'), index_col=0)
        reports = pd.read_csv(os.path.join(datadir, f'./{mode}/indiana_reports.csv'), index_col=0)

        # drop NaN findings and impressions
        not_null_idx = ~(reports['findings'].isnull() * reports['impression'].isnull())
        reports = reports[not_null_idx][self._report_sections_]
        df_frontal = projection[projection['projection']=='Frontal']
        df_lateral = projection[projection['projection']=='Lateral']

        self.uid2frontal = defaultdict(list)
        self.uid2lateral = defaultdict(list)

        for idx in reports.index.tolist():
            if idx in df_frontal.index:
                names = df_frontal.loc[idx].filename
                if isinstance(names, str): self.uid2frontal[idx].append(os.path.join(self.image_dir, names))
                else: self.uid2frontal[idx].extend([os.path.join(self.image_dir,name) for name in names.tolist()])
            if idx in df_lateral.index:
                names = df_lateral.loc[idx].filename
                if isinstance(names, str): self.uid2lateral[idx].append(os.path.join(self.image_dir, names))
                else: self.uid2lateral[idx].extend([os.path.join(self.image_dir,name) for name in names.tolist()])

        self.reports = reports.reset_index()
        # check if one report does have both frontal and lateral image
        f_uid_list = list(self.uid2frontal.keys())
        l_uid_list = list(self.uid2lateral.keys())
        x1 = [x for x in reports.index.tolist() if x not in f_uid_list]
        x2 = [x for x in reports.index.tolist() if x not in l_uid_list]
        print(np.intersect1d(x1, x2))

    def compute_img_mean_std(self):
        pixel_num  = 0
        channel_sum = np.zeros(self.channel_num)
        channel_sum_squared = np.zeros(self.channel_num)
        for index in self.reports.index.tolist():
            uid = self.reports.iloc[index].uid
            print('compute image mean and std, uid: ', uid)
            for filename in self.uid2frontal[uid]:
                x_image = read_image(filename) # 1, 2048, 2496
                img = x_image / 255
                pixel_num += torch.prod(torch.tensor(img.shape)).item()
                channel_sum += torch.sum(img).item()
                channel_sum_squared += torch.sum(img.square()).item()

            for filename in self.uid2lateral[uid]:
                x_image = read_image(filename)
                img = x_image / 255
                pixel_num += torch.prod(torch.tensor(img.shape)).item()
                channel_sum += torch.sum(img).item()
                channel_sum_squared += torch.sum(img.square()).item()

        img_mean = channel_sum / pixel_num
        img_std = np.sqrt(channel_sum_squared/pixel_num - np.square(img_mean))
        return {'mean':img_mean[0], 'std':img_std[0]}

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        '''return 
        1. list of frontal images
        2. list of lateral images
        3. the report texts
        4. the normal/abnormal label
        '''
        report = self.reports.iloc[idx]
        uid = report.uid
        report_str = ' '.join(report[self._report_sections_].fillna(' ').values.tolist())
        f_list = []
        for filename in self.uid2frontal[uid]:
            x_image = Image.open(filename)
            f_list.append(x_image)
        l_list = []
        for filename in self.uid2lateral[uid]:
            x_image = Image.open(filename)
            l_list.append(x_image)

        if len(f_list) == 0 and len(l_list) == 0:
            pdb.set_trace()
            pass
        return {'frontal': f_list, 'lateral': l_list, 'report': report_str, 'label': report['MeSH']}

class IUXRaySentenceDataset(Dataset):
    def __init__(self, datadir):
        import json
        self.datadir = datadir
        self.file_path = os.path.join(datadir, 'sentence_dict.txt')
        if not os.path.exists(self.file_path):
            print('no sentence file found, start to process')
            self._process()
        with open(self.file_path, 'r') as f:
            sent_dict = json.loads(f.read())
        self.sent_ts = pd.Series(sent_dict)

    def __len__(self):
        return len(self.sent_ts)

    def __getitem__(self, idx):
        sent = self.sent_ts.index[idx]
        return sent

    def _process(self):
        from tqdm import tqdm
        import json
        df = pd.read_csv(os.path.join(self.datadir,'indiana_reports.csv'))
        df_reports = df[['findings','impression']]
        df_reports.fillna('', inplace=True)
        df_sentences = df_reports.applymap(lambda x: x.split('.'))
        all_sent_dict = defaultdict(int)
        for sample in tqdm(df_sentences.values):
            for sent_list in sample:
                for sent in sent_list:
                    if len(sent)>0: all_sent_dict[sent.strip().lower()]+=1     
        with open(self.file_path,'w') as f:
            f.write(json.dumps(all_sent_dict))

# ########
# Three collators for three contrastive loss computation
# ########
class IUXRayCollatorBase:
    def __init__(self,
        feature_extractor=None,
        tokenizer=None,
        img_mean=None,
        img_std=None,
        max_text_length=77,
        set_tokenizer=True,
        set_feature_extractor=True,
        ):
        '''args:
        set_tokenizer: set True if need tokenizer
        set_feature_extractor: set True if need image feature extractor
        '''
        if feature_extractor is None and set_feature_extractor:
            assert img_mean is not None
            assert img_std is not None
            self.feature_extractor = MedCLIPFeatureExtractor(
                do_resize=True,
                size=224,
                resample=3,
                do_center_crop=True,
                crop_size=224,
                do_normalize=True,
                image_mean=img_mean,
                image_std=img_std,
            )
        else:
            self.feature_extractor = feature_extractor

        if tokenizer is None and set_tokenizer:
            self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        elif isinstance(tokenizer, str):
            if os.path.exists(tokenizer):
                self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer)
            else:
                self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
                self.tokenizer.save_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        self.tokenizer.model_max_length = max_text_length
    
    def __call__(self, x):
        raise NotImplementedError

class IUXRayImageTextCollator(IUXRayCollatorBase):
    def __init__(self, feature_extractor=None, tokenizer=None, img_mean=None, img_std=None, is_train=False):
        '''return image-text report positive pairs
        '''
        super().__init__(feature_extractor, tokenizer, img_mean, img_std)
        self.is_train = is_train

    def __call__(self, x):
        '''
        x: list of dict{frontal, lateral, report, label}
        return {'input_ids': [], 'pixel_values': [], 'attention_mask':[], 'report':[]}
        '''
        inputs = defaultdict(list)
        text_list = []
        for data in x: # every data is a single patient
            report = data['report']
            if self.is_train: report = self._text_random_cut_(report)
            if len(data['frontal']) > 0:
                images = self.feature_extractor(data['frontal'], return_tensors='pt')
                inputs['pixel_values'].append(images['pixel_values'])
                text_list.extend([report] * len(data['frontal']))
            if len(data['lateral']) > 0:
                images = self.feature_extractor(data['lateral'], return_tensors='pt')
                inputs['pixel_values'].append(images['pixel_values'])
                text_list.extend([report] * len(data['lateral']))
        # tokenize texts together
        try:
            text_token_ids = self.tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
        except:
            pdb.set_trace()

        for key in text_token_ids.keys():
            inputs[key] = text_token_ids[key]
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'])
        if not self.is_train: inputs['report'] = text_list
        return inputs

    def _text_random_cut_(self, text):
        token_list = text.split(' ')
        max_start_idx = np.maximum(len(token_list)-self.tokenizer.model_max_length, 0)
        if max_start_idx == 0:
            return text
        else:
            start_idx = np.random.randint(0, max_start_idx)
            return ' '.join(token_list[start_idx:start_idx+self.tokenizer.model_max_length])

class IUXRayAbnormalNormalCollator(IUXRayCollatorBase):
    def __init__(self, feature_extractor=None, tokenizer=None, img_mean=None, img_std=None, is_train=False):
        '''return abnormal-normal positive pairs,
        normal: label 0
        abnormal: label 1
        '''
        super().__init__(feature_extractor, tokenizer, img_mean, img_std)
        self.is_train = is_train

    def __call__(self, x):
        '''
        x: list of dict{frontal, lateral, report, label}
        return {'pixel_values':[], 'labels':[]}
        abnormals encodings will be saved in an momentum memory bank
        '''
        inputs = defaultdict(list)
        for data in x:
            label = 0 if data['label']=='normal' else 1
            num_images = len(data['frontal']) + len(data['lateral'])
            if len(data['frontal']) > 0:
                images = self.feature_extractor(data['frontal'], return_tensors='pt')
                inputs['pixel_values'].append(images['pixel_values'])
            if len(data['lateral']) > 0:
                images = self.feature_extractor(data['lateral'], return_tensors='pt')
                inputs['pixel_values'].append(images['pixel_values'])
            inputs['labels'].extend([label]*num_images)
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'])
        inputs['labels'] = torch.tensor(inputs['labels'])
        return inputs

class IUXRayFrontalLateralCollator(IUXRayCollatorBase):
    def __init__(self, feature_extractor=None, tokenizer=None, img_mean=None, img_std=None, is_train=False):
        '''return frontal-lateral positive pairs,
        0: frontal
        1: lateral
        '''
        super().__init__(feature_extractor, tokenizer, img_mean, img_std)
        self.is_train = is_train

    def __call__(self, x):
        '''
        x: list of dict{frontal, lateral, report, label}
        return {'pixel_values':[], 'labels':[]}
        labels==0: frontal
        labels==1: lateral
        '''
        inputs = defaultdict(list)
        for data in x:
            if len(data['frontal']) > 0:
                images = self.feature_extractor(data['frontal'], return_tensors='pt')
                inputs['pixel_values'].append(images['pixel_values'])
                inputs['labels'].extend([0]*len(data['frontal']))
            if len(data['lateral']) > 0:
                images = self.feature_extractor(data['lateral'], return_tensors='pt')
                inputs['pixel_values'].append(images['pixel_values'])
                inputs['labels'].extend([1]*len(data['lateral']))
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'])
        inputs['labels'] = torch.tensor(inputs['labels'])
        return inputs

class IUXRayTextCollator(IUXRayCollatorBase):
    def __init__(self, tokenizer='./medclip/cliptokenizer'):
        super().__init__(tokenizer=tokenizer, set_feature_extractor=False)
    
    def __call__(self, x):
        '''
        x: list of sentences
        '''
        output = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')
        output['sentence'] = x
        return output

        
