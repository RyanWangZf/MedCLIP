import re
import random
from collections import defaultdict
import pdb

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder


from .prompts import process_class_prompts
from .prompts import generate_chexpert_class_prompts
from . import constants

# TODO
# 1. mixup?
# 2. autoaugment?

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

        # could try contrast, brightness, fog
        if imgtransform is None:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        # use labeled sentences as prompts for chexpert training
        self.sentence_label = pd.read_csv('./local_data/sentence-label.csv', index_col=0).fillna(0)
        print('load sentence prompts from ./local_data/sentence-label.csv')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)
        img = self._pad_img(img) # pad image to square
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
            # we need to use sentence-level label instead
            # maintain a sentence dictionary
            # index sentence dictionary label during training, if not found, return all zero
            # **supervision from unpaired texts and images**
            if report in self.sent_label_dict: # find the sentence
                text_label = self.sent_label_dict[report]
            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1
        return img, report, img_label, text_label
            
    def __len__(self):
        return len(self.df)
    
    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im
    
    def sample_sent_prompts(self, row):
        # do prompt sampling
        if (row[self._labels_] == 0).all(): # no label available, use no finding
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
            bool_sent_label = self.prompt_sentence_label[self._labels_] *  row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: # only no finding
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                # random sample
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
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
                if len(included_tokens) > 4: # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(subset='Reports')
        self.sentence_label = self.sentence_label[self.sentence_label['Reports'].map(len)>2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Reports']
        self.sentence_label = self.sentence_label.drop('Reports', axis=1)
        self.sentence_label = self.create_sent_segments(self.sentence_label)
        self.sentence_label = self.sentence_label[(self.sentence_label['report'].map(len)==1)]
        self.sentence_label['report'] = np.concatenate(self.sentence_label['report'].values)
        # build dict from dataframe
        keys = self.sentence_label['report'].values
        vals = self.sentence_label.drop(['report'],axis=1).fillna(0).values
        self.sent_label_dict = dict(zip(keys,vals))
    
    def _build_prompt_sentence(self, n = 200):
        print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        for task in constants.CHEXPERT_TASKS:
            sub_sent_df = sentence_label[sentence_label[task] == 1]
            if len(sub_sent_df) < n: new_sent_list.append(sub_sent_df)
            else: new_sent_list.append(sub_sent_df.sample(n))

        new_sent_df = pd.concat(new_sent_list, 0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        '''Args:
        use_EDA: easy data augmentation from textaugment
        '''
        if use_eda:
            import nltk
            nltk.download('stopwords')
            nltk.download('omw-1.4')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0])
            if self.eda is not None:
                eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                text_aug = eda_aug(data[1])
                if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                report_aug_list.append(text_aug)
            report_list.append(data[1])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] =  aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']
            
        return inputs

class ZeroShotImageDataset(Dataset):
    def __init__(self, 
        datalist=['chexpert-5x200'],
        class_names=None,
        imgtransform=None,
        ) -> None:
        '''support data list in iuxray, mimic-cxr, chexpert, chexpert-5x200;
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
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
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

        if cls_prompts is None:
            self.cls_prompts = generate_chexpert_class_prompts(n=n_prompt)
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
        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'], 
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
            }

class SuperviseImageDataset(Dataset):
    def __init__(self, 
        datalist=['chexpert-5x200'],
        class_names=None,
        imgtransform=None, 
        ) -> None:
        '''support data list in iuxray, mimic-cxr, chexpert, chexpert, covid19;
        args:
            imgtransform: a torchvision transform
            class_names: a list of class names
        '''
        super().__init__()
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
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

class SuperviseImageCollator:
    def __init__(self, mode):
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])
        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values

        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'], 
            'labels': inputs['labels'],
            }
