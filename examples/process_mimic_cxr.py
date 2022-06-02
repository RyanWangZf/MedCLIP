import pdb
import os
import pandas as pd
import numpy as np
import random

from medclip import constants

np.random.seed(42)
random.seed(42)

src_dir = './data/MIMIC-CXR/'
tgt_dir = './local_data'

if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

meta_path = os.path.join(src_dir, 'mimic-cxr-2.0.0-metadata.csv')
df_meta = pd.read_csv(meta_path)
df_label = pd.read_csv(os.path.join(src_dir, 'mimic-cxr-2.0.0-negbio.csv'))
df_merge = df_meta.merge(df_label, on ='study_id', how='left')
df_merge['imgpath'] = df_merge['dicom_id'].apply(lambda x: os.path.join(src_dir, 'images' ,f'{x}.jpg'))
df_report = pd.read_csv(os.path.join(src_dir, 'mimic_finding+impression2.csv'))
columns = [
    'study_id', 'imgpath', 'ViewPosition', 'subject_id_x',
]
columns += constants.CHEXPERT_TASKS
df_merge = df_merge[columns]
df_merge = df_merge.rename(columns={'subject_id_x':'subject_id'})
df_merge = df_merge.fillna(0)
df_merge = df_merge.replace({-1:0})
df_merge['report'] = df_report['finding'].fillna('') + df_report['impression'].fillna('')

df_merge.to_csv(
    os.path.join(tgt_dir, 'mimic-cxr-meta.csv')
)
df = df_merge

# get mimic-5x200 finetune and mimic-5x200 test
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
df_200.to_csv(f'{tgt_dir}/mimic-5x200-meta.csv')
df.to_csv(f'{tgt_dir}/mimic-cxr-train-meta.csv')
df_finetune = df_400[~df_400['imgpath'].isin(df_200['imgpath'])]
df_finetune.to_csv(f'{tgt_dir}/mimic-5x200-finetune-meta.csv')
