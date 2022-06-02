# split train/test
import pdb
import os
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

src_dir = './data/CheXpert/CheXpert-v1.0-small/'
tgt_dir = './local_data'

if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)
meta_path = os.path.join(src_dir, 'train.csv')
df = pd.read_csv(meta_path)
df['subject_id'] = df['Path'].apply(lambda x: '-'.join(x.split('/')[2:4]))
df = df.rename(columns = {'Path':'imgpath'})
df = df.fillna(0)
df = df.replace({-1:0})

# get chexpert5x200 and chexpert-5x200-finetune
from medclip.constants import CHEXPERT_COMPETITION_TASKS
task_dfs = []
for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
    index = np.zeros(14)
    index[i] = 1
    df_task = df[
        (df["Atelectasis"] == index[0])
        & (df["Cardiomegaly"] == index[1])
        & (df["Consolidation"] == index[2])
        & (df["Edema"] == index[3])
        & (df["Pleural Effusion"] == index[4])
        & (df["Enlarged Cardiomediastinum"] == index[5])
        & (df["Lung Lesion"] == index[7])
        & (df["Lung Opacity"] == index[8])
        & (df["Pneumonia"] == index[9])
        & (df["Pneumothorax"] == index[10])
        & (df["Pleural Other"] == index[11])
        & (df["Fracture"] == index[12])
        & (df["Support Devices"] == index[13])
    ]
    df_task = df_task.sample(n=400)
    task_dfs.append(df_task)

df_400 = pd.concat(task_dfs)
df_200 = df_400.sample(frac=0.5)
df = df[~df['imgpath'].isin(df_200['imgpath'])]
df_200.to_csv(f'{tgt_dir}/chexpert-5x200-meta.csv')
df.to_csv(f'{tgt_dir}/chexpert-train-meta.csv')
df_finetune = df_400[~df_400['imgpath'].isin(df_200['imgpath'])]
df_finetune.to_csv(f'{tgt_dir}/chexpert-5x200-finetune-meta.csv')
