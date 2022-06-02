import pdb
from collections import defaultdict
import os
import pandas as pd
import numpy as np
import random
import pydicom
from tqdm import tqdm
from PIL import Image

np.random.seed(42)
random.seed(42)

src_dir = './data/RSNA'
tgt_dir = './local_data'


if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)
dst_dir = os.path.join(src_dir, 'images')
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
src_img_dir = os.path.join(src_dir, 'stage_2_train_images/')
meta_df = defaultdict(list)

src_files = os.listdir(src_img_dir)
for filename in tqdm(src_files):
    src_path = os.path.join(src_img_dir, filename)
    x = pydicom.read_file(src_path)
    pixels = x.pixel_array
    view_position = x.ViewPosition
    meta_df['view'].append(view_position)
    dst_filename = filename.replace('.dcm','.png')
    dst_path = os.path.join(dst_dir, dst_filename)
    img = Image.fromarray(pixels)
    img.save(dst_path)
    meta_df['imgpath'].append(dst_path)
    img_id = filename.replace('.dcm','')
    meta_df['id'].append(img_id)

df = pd.DataFrame(meta_df)
df.to_csv(os.path.join(tgt_dir, 'rsna-meta.csv'))
df = pd.read_csv(os.path.join(src_dir, './stage_2_train_labels.csv')).rename(columns={'patientId':'id'})
df_meta = pd.read_csv(os.path.join(tgt_dir, './rsna-meta.csv'), index_col=0)
df_merge = df_meta.merge(df[['id','Target']], on='id').drop_duplicates()
pneumonia = np.zeros(len(df_merge))
pneumonia[df_merge['Target']==1] = 1
df_merge['Pneumonia'] = pneumonia
normal = np.zeros(len(df_merge))
normal[df_merge['Target']==0] = 1
df_merge['Normal'] = normal
df_merge.drop(['Target'], axis=1).to_csv(os.path.join(tgt_dir, './rsna-meta.csv'))

df_meta = pd.read_csv(os.path.join(tgt_dir, './rsna-meta.csv'), index_col=0)
df_train = df_meta.sample(frac=0.7)
df_test = df_meta[~df_meta['imgpath'].isin(df_train['imgpath'])]
df_train.to_csv(os.path.join(tgt_dir,'rsna-train-meta.csv'))
df_test.to_csv(os.path.join(tgt_dir,'rsna-test-meta.csv'))
