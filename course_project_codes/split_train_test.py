'''
split IUXray into train/val/test
as 0.7/0.1/0.2
'''
import os, pdb
import shutil
import pandas as pd
import numpy as np
np.random.seed(42)

# #####################
# define output dirs
datadir = './data/IU_XRay'
image_dir = os.path.join(datadir, './images/images_normalized')
reports = pd.read_csv(os.path.join(datadir, 'indiana_reports.csv'), index_col=0)
projection = pd.read_csv(os.path.join(datadir, 'indiana_projections.csv'), index_col=0)

# define output dirs
out_train_dir = './data/IU_XRay/train'
if not os.path.exists(out_train_dir): os.makedirs(out_train_dir)
out_val_dir = './data/IU_XRay/val'
out_test_dir = './data/IU_XRay/test'

# remove NaN findings and impressions
not_null_idx = ~(reports['findings'].isnull() * reports['impression'].isnull())
reports = reports[not_null_idx]

all_idx = np.arange(len(reports))
np.random.shuffle(all_idx)
num_train = int(len(reports) * 0.7)
train_idx = all_idx[:num_train]
num_val = int(len(reports) * 0.1)
val_idx = all_idx[num_train:num_train+num_val]
num_test = len(reports) - num_train - num_val
test_idx = all_idx[num_train+num_val:]

def func(out_dir, set_idx):
    print('start processing')
    reports_sub = reports.iloc[set_idx]
    out_report_path = os.path.join(out_dir, 'indiana_reports.csv')
    projection_sub = projection.loc[reports_sub.index]
    out_proj_path = os.path.join(out_dir, 'indiana_projections.csv')
    out_image_dir = os.path.join(out_dir, 'images/images_normalized')
    if not os.path.exists(out_image_dir): os.makedirs(out_image_dir)
    for i in range(len(projection_sub)):
        if i % 100 == 0:
            print(f'{out_dir} {i}/{len(projection_sub)}')
        filename = projection_sub.iloc[i]['filename']
        src_path = os.path.join(image_dir, filename)
        dest_path = os.path.join(out_image_dir, filename)
        shutil.copy(src_path, dest_path)
    reports_sub.to_csv(out_report_path)
    projection_sub.to_csv(out_proj_path)

func(out_train_dir, train_idx)
func(out_val_dir, val_idx)
func(out_test_dir, test_idx)
print('done')