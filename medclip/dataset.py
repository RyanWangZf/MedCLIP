import os
import re
import random
from collections import defaultdict
import pdb
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms

from transformers import AutoTokenizer
from transformers import CLIPFeatureExtractor, CLIPProcessor
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor

import nltk
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder

from .prompts import process_class_prompts, process_class_prompts_for_tuning
from .prompts import generate_chexpert_class_prompts
from . import constants

class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, 
        do_resize=True, 
        size=224, 
        resample=Image.BICUBIC, 
        do_center_crop=True, 
        crop_size=224, 
        do_normalize=True, 
        image_mean=constants.IMG_MEAN, 
        image_std=constants.IMG_STD, 
        do_convert_rgb=False,
        do_pad_square=True,
        **kwargs):
        super().__init__(do_resize, size, resample, do_center_crop, crop_size, do_normalize, image_mean, image_std, do_convert_rgb, **kwargs)
        self.do_pad_square = do_pad_square
    
    def __call__(self, 
        images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]], 
        return_tensors: Optional[Union[str, TensorType]] = None, 
        **kwargs) -> BatchFeature:
        
        valid_images = False

        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]

        if self.do_pad_square:
            images = [self.pad_img(image,min_size=self.size) for image in images]
        
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        images = images_

        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

class MedCLIPProcessor(CLIPProcessor):
    feature_extractor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    def __init__(self):
        feature_extractor = MedCLIPFeatureExtractor()
        tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        tokenizer.model_max_length = 77
        super().__init__(feature_extractor, tokenizer)

class ImageTextContrastiveDataset(Dataset):
    _labels_ = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    def __init__(self, datalist=['mimic-cxr-train', 'chexpert-train'], imgtransform=None) -> None:
        super().__init__()
        df_list = []
        for data in datalist:
            filename = os.path.join('.', 'local_data', f'{data}-meta.csv')
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

        self.df = self.create_sent_segments(self.df)

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

        prompt_path = os.path.join('.', 'local_data', 'sentence-label.csv')
        self.sentence_label = pd.read_csv(prompt_path, index_col=0).fillna(0)
        print(f'load sentence prompts from {prompt_path}')
        self._preprocess_sentence_label()
        self._build_prompt_sentence()

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row.imgpath)

        img = self._pad_img(img) 
        img = self.transform(img).unsqueeze(1)
        report = row.report 
        img_label = row[self._labels_].values 
        if len(report) == 0: 
            report, text_label = self.sample_sent_prompts(row)
        else:
            sent_ix = random.randint(0, len(report)-1)
            report = report[sent_ix]
            if report in self.sent_label_dict: 
                text_label = self.sent_label_dict[report]
            else:
                text_label = np.zeros(len(img_label))
                text_label[0] = 1
        return img, report, img_label, text_label

    def __len__(self):
        return len(self.df)

    def _pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def sample_sent_prompts(self, row):
        if (row[self._labels_] == 0).all(): 
            sampled_sent = self.sentence_label[self.sentence_label['No Finding'] > 0].sample()
            report = sampled_sent['report'].values[0][0]
            label = sampled_sent[self._labels_].values[0]
        else:
            bool_sent_label = self.prompt_sentence_label[self._labels_] * row[self._labels_]
            bool_sent_label[bool_sent_label < 0] = 0
            sents = self.prompt_sentence_label.loc[~(bool_sent_label.iloc[:,1:] == 0).all(1)]
            if len(sents) == 0: 
                sampled_sent = self.prompt_sentence_label[self.prompt_sentence_label['No Finding']==1].sample()
            else:
                sampled_sent = sents.sample()
            report = sampled_sent['report'].values[0]
            label = sampled_sent[self._labels_].values.flatten()
        return report, label

    def create_sent_segments(self, df):
        df['report'] = df['report'].apply(self._split_report_into_segment)
        return df

    def _split_report_into_segment(self, report):
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n',' ')
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                tokens = nltk.wordpunct_tokenize(sent.lower())
                
                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4: 
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

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        self.prompt_sentence_label = new_sent_df

class ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        if use_eda:
            import nltk
            nltk.download('stopwords')
            nltk.download('omw-1.4')
            nltk.download('wordnet')
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

        df_list = []
        for data in datalist:
            filename = os.path.join('.', 'local_data', f'{data}-meta.csv')
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
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

class ZeroShotImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5):
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

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

        df_list = []
        for data in datalist:
            filename = os.path.join('.', 'local_data', f'{data}-meta.csv')
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

class PromptTuningImageDataset(Dataset):
    def __init__(self,
                 datalist=['chexpert-5x200'],
                 class_names=None,
                 imgtransform=None,
                 ) -> None:
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

        df_list = []
        for data in datalist:
            filename = os.path.join('.', 'local_data', f'{data}-meta.csv')
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
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class PromptTuningImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5, n_context=16, class_specific_context=False):
        assert mode in ['multiclass', 'multilabel', 'binary']
        self.mode = mode

        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts

        self.prompt_texts_inputs = process_class_prompts_for_tuning(self.cls_prompts,
                                                                    n_context=n_context,
                                                                    class_specific_context=class_specific_context)

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