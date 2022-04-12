import pandas as pd
import numpy as np
import pdb

# IUXRAY
# df = pd.read_csv('./local_data/iuxray-meta.csv', index_col=0)
# df_report_label = pd.read_csv('./local_data/iuxray-report-label.csv')
# df_report_label = df_report_label.fillna(0)

# df_new = df.merge(df_report_label, on='uid')
# df_new = df_new.drop(['projection','Reports','label'],axis=1)
# df_new = df_new.rename(columns={'uid':'subject_id'})
# df_new.to_csv('./local_data/iuxray-meta.csv')

# MIMIC
# df = pd.read_csv('./local_data/mimic-cxr-meta.csv', index_col=0)
# df.drop(['projection'], axis=1).to_csv('./local_data/mimic-cxr-meta.csv')

# Chexpert
# df = pd.read_csv('./local_data/chexpert-meta.csv', index_col=0)
# df.rename(columns={'study_id':'subject_id'}).to_csv('local_data/chexpert-meta.csv')

# get chexpert-5-200 data for zero-shot classification
# from medclip.constants import CHEXPERT_COMPETITION_TASKS
# df = pd.read_csv('./local_data/chexpert-meta.csv', index_col=0)
# df = df.fillna(0)
# task_dfs = []
# for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
#     index = np.zeros(14)
#     index[i] = 1
#     df_task = df[
#         (df["Atelectasis"] == index[0])
#         & (df["Cardiomegaly"] == index[1])
#         & (df["Consolidation"] == index[2])
#         & (df["Edema"] == index[3])
#         & (df["Pleural Effusion"] == index[4])
#         & (df["Enlarged Cardiomediastinum"] == index[5])
#         & (df["Lung Lesion"] == index[7])
#         & (df["Lung Opacity"] == index[8])
#         & (df["Pneumonia"] == index[9])
#         & (df["Pneumothorax"] == index[10])
#         & (df["Pleural Other"] == index[11])
#         & (df["Fracture"] == index[12])
#         & (df["Support Devices"] == index[13])
#     ]
#     df_task = df_task.sample(n=200)
#     task_dfs.append(df_task)

# df_200 = pd.concat(task_dfs)
# df = df[~df['imgpath'].isin(df_200['imgpath'])]
# df_200.to_csv('./local_data/chexpert-5x200.csv')
# df.to_csv('./local_data/chexpert-train-meta.csv')
