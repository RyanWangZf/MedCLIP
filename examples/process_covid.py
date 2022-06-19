import pdb
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname("__file__")))
print(sys.path)
np.random.seed(42)
random.seed(42)

data_dir = './data/COVID19/COVID-19_Radiography_Dataset/'
tgt_dir = './local_data'

tasks = ['Normal', 'Lung_Opacity','COVID',]
df_meta_list = []
for task in tasks:
    df = pd.read_excel(os.path.join(data_dir, f'{task}.metadata.xlsx'))
    if task == 'Lung_Opacity':
        df['FILE NAME'] = df['FILE NAME'].apply(lambda x: '_'.join([xx.capitalize() for xx in x.split('_')]))
    elif task == 'Normal':
        df['FILE NAME'] = df['FILE NAME'].apply(lambda x: x.capitalize())

    img_dir = os.path.join(data_dir, f'{task}/images/')

    df['imgpath'] = df['FILE NAME'].apply(lambda x: os.path.join(img_dir, f'{x}.png'))
    for t in tasks:
        if t == task:
            df[t] = np.ones(len(df))
        else:
            df[t] = np.zeros(len(df))
    df_meta_list.append(df[tasks+['imgpath']])
df_meta = pd.concat(df_meta_list).reset_index(drop=True)
df_meta = df_meta.rename(columns={'Lung_Opacity':'Lung Opacity'})
df_bin = df_meta.drop(['Lung Opacity'], axis=1)
df_bin = df_bin[df_bin[['Normal','COVID']].sum(1) > 0].reset_index(drop=True)
df_cv = df_bin[df_bin['COVID'] == 1]
df_normal = df_bin[df_bin['Normal']==1].sample(len(df_cv))
df_test = pd.concat([df_cv,df_normal],axis=0).reset_index(drop=True)
df_test = df_test.sample(n=3000)
df_train = df_bin[~df_bin['imgpath'].isin(df_test['imgpath'])]
df_train.to_csv(f'{tgt_dir}/covid-train-meta.csv')
df_test.to_csv(f'{tgt_dir}/covid-test-meta.csv')
df_train.sample(frac=0.1).to_csv(f'{tgt_dir}/covid-0.1-train-meta.csv')
df_train.sample(frac=0.2).to_csv(f'{tgt_dir}/covid-0.2-train-meta.csv')
