import pandas as pd
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
df = pd.read_csv('./local_data/chexpert-meta.csv', index_col=0)
df.rename(columns={'study_id':'subject_id'}).to_csv('local_data/chexpert-meta.csv')
