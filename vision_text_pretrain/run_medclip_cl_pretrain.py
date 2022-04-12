import pdb, os
import math
import random
from collections import defaultdict
import re

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer

os.environ['CUDA_VISIBLE_DEVICES']='1'

from medclip.modeling_medclip import MedClipModel
from medclip.trainer import Trainer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_config = {
    'batch_size': 128,
    'num_epochs': 10,
    'warmup': 0.01, # the first 1% of training steps are used for warm-up
    'lr': 5e-5,
    'weight_decay': 1e-2,
    'eval_batch_size': 128,
    'eval_steps': 100,
    'save_steps': 500,
}

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

trainloader = DataLoader(traindata, 
    batch_size=train_config['batch_size'], collate_fn=collate_fn, 
    shuffle=True,
    pin_memory=False,
    num_workers=0,
    )

# model = MedClipModel(
#     vision_checkpoint='./checkpoints/vision_pretrain'
#     )

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








