import re
import random
from collections import defaultdict
import pdb

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
from PIL import Image

from .prompts import process_class_prompts
from .prompts import generate_chexpert_class_prompts
from . import constants


class ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datalist=['chexpert', 'mimic-cxr', 'iuxray'], imgtransform=None) -> None:
        '''support data list in iuxray, mimic-cxr, chexpert
        '''
        super().__init__()
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        # split raw reports and process into sentences
        self.df = self.create_sent_segments(self.df)

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('./local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[~(self.sentence_label['report'].map(len)==0)]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self.transform(img).unsqueeze(1)
        report = row.report # original sentences list
        img_label = row[self._labels_].values # image corresponds to text labels
        if len(report) == 0: # no report available
            # sample class prompts as augmentation
            report, text_label = self.sample_sent_prompts(row)
        else:
            # randomly sample one sentence
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
            # TODO: now we simply take one sentence from the whole report same label as the report
            # we need to use sentence-level label instead
            text_label = img_label
        return img, report, img_label, text_label
            
    def __len__(self):
        return len(self.df)
    
    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.sentence_label[~(bool_sent_label == 0).all(1)].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

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
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        return inputs

class ZeroShotImageDataset(Dataset):
    def __init__(self, datalist=['chexpert-5x200'], imgtransform=None) -> None:
        '''support data list in iuxray, mimic-cxr, chexpert, chexpert-5x200;
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform
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
        img = self.transform(img).unsqueeze(1).repeat((1,3,1,1)) # align with resnet50 3-channel inputs
        label = row[row == 1].index[0]
        return img, label
    
    def __len__(self):
        return len(self.df)

class ZeroShotImageCollator:
    def __init__(self, cls_prompts=None, n_prompt=5):
        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.tokenizer.model_max_length = 77

        if cls_prompts is None:
            self.cls_prompts = generate_chexpert_class_prompts(n=n_prompt)
        else:
            self.cls_prompts = cls_prompts

        # process cls prompts into texts indices
        self.prompt_texts_inputs =  process_class_prompts(self.cls_prompts)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        return {
            'pixel_values': inputs['pixel_values'], 
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
            }