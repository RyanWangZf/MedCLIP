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
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform

        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('./local_data/iuxray-sentence-label.csv').fillna(0)
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)

        # get negative phrase sentences
        self.negative_sent_label = self.sentence_label.loc[(self.sentence_label[self._labels_] == -1).sum(1) > 0].copy()

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('phdf33/trialbert-base')
        self.tokenizer.model_max_length = 77

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self.transform(img).unsqueeze(1)
        report = row.report # original sentences list

        if len(report) == 0: # no report available
            # sample class prompts as augmentation
            report = self.sample_sent_prompts(row)
        else:
            # randomly sample one sentence
            sent_ix = random.randint(0, len(report))
            report = report[sent_ix]

        return img, report
            
    def __len__(self):
        return len(self.df)
    
    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
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
        return report

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

    def collate_fn(self, batch):
        inputs = defaultdict(list)
        report_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0])
            report_list.append(data[1])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        return inputs


class ZeroShotImageDataset(Dataset):
    def __init__(self, datalist=['chexpert-5x200'], imgtransform=None, cls_prompts=None) -> None:
        '''support data list in iuxray, mimic-cxr, chexpert, chexpert-5x200;
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()
        if cls_prompts is None:
            from .prompts import generate_chexpert_class_prompts
            cls_prompts = generate_chexpert_class_prompts()
        else:
            cls_prompts = cls_prompts
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256,256)),
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

        # initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('phdf33/trialbert-base')
        self.tokenizer.model_max_length = 77

        # process cls prompts into texts indices
        self.prompt_texts_inputs =  self.process_class_prompts(cls_prompts)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self.transform(img).unsqueeze(1)
        label = row[row == 1].index[0]
        return img, label
    
    def __len__(self):
        return len(self.df)

    def process_class_prompts(self, cls_prompts):
        cls_prompt_inputs = defaultdict()
        for k,v in cls_prompts.items():
            text_inputs = self.tokenizer(v, truncation=True, padding=True, return_tensors='pt')
            cls_prompt_inputs[k] = text_inputs
        return cls_prompt_inputs

    def collate_fn(self, batch):
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