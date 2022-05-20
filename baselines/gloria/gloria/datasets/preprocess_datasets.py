import pandas as pd
import numpy as np
import os
import sys
import tqdm
import argparse

sys.path.append(os.getcwd())
from gloria.constants import *
from sklearn.model_selection import train_test_split


def preprocess_pneumonia_data(test_fac=0.15):

    try:
        df = pd.read_csv(PNEUMONIA_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the RSNA Pneumonia dataset is \
            stored at {PNEUMONIA_DATA_DIR}"
        )

    # create bounding boxes
    def create_bbox(row):
        if row["Target"] == 0:
            return 0
        else:
            x1 = row["x"]
            y1 = row["y"]
            x2 = x1 + row["width"]
            y2 = y1 + row["height"]
            return [x1, y1, x2, y2]

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # no encoded pixels mean healthy
    df["Path"] = df["patientId"].apply(lambda x: PNEUMONIA_IMG_DIR / (x + ".dcm"))

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    train_df.to_csv(PNEUMONIA_TRAIN_CSV)
    valid_df.to_csv(PNEUMONIA_VALID_CSV)
    test_df.to_csv(PNEUMONIA_TEST_CSV)


def preprocess_pneumothorax_data(test_fac=0.15):

    try:
        df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the SIIM Pneumothorax dataset is \
            stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    # get image paths
    img_paths = {}
    for subdir, dirs, files in tqdm.tqdm(os.walk(PNEUMOTHORAX_IMG_DIR)):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                img_paths[file_id] = os.path.join(subdir, f)

    # no encoded pixels mean healthy
    df["Label"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths[x])

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Label"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Label"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Label"].value_counts())

    train_df.to_csv(PNEUMOTHORAX_TRAIN_CSV)
    valid_df.to_csv(PNEUMOTHORAX_VALID_CSV)
    test_df.to_csv(PNEUMOTHORAX_TEST_CSV)


def preprocess_chexpert_5x200_data():

    df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    df = df.fillna(0)
    df = df[df["Frontal/Lateral"] == "Frontal"]

    df_master = pd.read_csv(CHEXPERT_MASTER_CSV)
    df_master = df_master[["Path", "Report Impression"]]

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
        df_task = df_task.sample(n=200)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)

    # get reports
    df_200 = pd.merge(df_200, df_master, how="left", left_on="Path", right_on="Path")

    return df_200


def preprocess_chexpert_data():

    try:
        df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the Pneunotrhoax dataset is \
            stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    df_200 = preprocess_chexpert_5x200_data()
    df = df[~df[CHEXPERT_PATH_COL].isin(df_200[CHEXPERT_PATH_COL])]
    valid_ids = np.random.choice(len(df), size=CHEXPERT_VALID_NUM, replace=False)
    valid_df = df.iloc[valid_ids]
    train_df = df.drop(valid_ids, errors="ignore")

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")
    print(f"Number of chexpert5x200 samples: {len(df_200)}")

    train_df.to_csv(CHEXPERT_TRAIN_CSV)
    valid_df.to_csv(CHEXPERT_VALID_CSV)
    df_200.to_csv(CHEXPERT_5x200)


_DATASETS = {
    "chexpert": preprocess_chexpert_data,
    "pneumonia": preprocess_pneumonia_data,
    "pneumothorax": preprocess_pneumothorax_data,
}


def available_datasets():
    """Returns the names of available datasets"""
    return list(_DATASETS.keys())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset type, one of [chexpert, pneumonia, pneumothorax]",
        required=True,
    )
    args = parser.parse_args()

    if args.dataset.lower() in _DATASETS.keys():
        _DATASETS[args.dataset.lower()]()
    else:
        RuntimeError(
            f"Model {args.dataset} not found; available datasets = {available_datasets()}"
        )
