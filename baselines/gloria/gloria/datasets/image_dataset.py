import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from albumentations import ShiftScaleRotate, Normalize, Resize, Compose
from albumentations.pytorch import ToTensor
from gloria.constants import *


class ImageBaseDataset(Dataset):
    def __init__(
        self,
        cfg,
        split="train",
        transform=None,
    ):

        self.cfg = cfg
        self.transform = transform
        self.split = split

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_jpg(self, img_path):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img

    def read_from_dicom(self, img_path):
        raise NotImplementedError

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


class CheXpertImageDataset(ImageBaseDataset):
    def __init__(self, cfg, split="train", transform=None, img_type="Frontal"):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(CHEXPERT_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(CHEXPERT_VALID_CSV)
        else:
            self.df = pd.read_csv(CHEXPERT_TEST_CSV)

        # sample data
        if cfg.data.frac != 1 and split == "train":
            self.df = self.df.sample(frac=cfg.data.frac)

        # filter image type
        if img_type != "All":
            self.df = self.df[self.df[CHEXPERT_VIEW_COL] == img_type]

        # get path
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        )

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
        super(CheXpertImageDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # get image
        img_path = row["Path"]
        x = self.read_from_jpg(img_path)

        # get labels
        y = list(row[CHEXPERT_COMPETITION_TASKS])
        y = torch.tensor(y)

        return x, y

    def __len__(self):
        return len(self.df)


class PneumothoraxImageDataset(ImageBaseDataset):
    def __init__(self, cfg, split="train", transform=None):

        if PNEUMOTHORAX_DATA_DIR is None:
            raise RuntimeError(
                "SIIM Pneumothorax data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview"
                + f" and update PNEUMOTHORAX_DATA_DIR in ./gloria/constants.py"
            )

        self.split = split
        self.cfg = cfg

        if cfg.phase == "segmentation":
            transform = None
            self.seg_transform = self.get_transforms()
        else:
            self.transform = transform

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(PNEUMOTHORAX_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(PNEUMOTHORAX_VALID_CSV)
        else:
            self.df = pd.read_csv(PNEUMOTHORAX_TEST_CSV)

        # only keep positive samples for segmentation
        self.df["class"] = self.df[" EncodedPixels"].apply(lambda x: x != " -1")
        if cfg.phase == "segmentation" and split == "train":
            self.df_neg = self.df[self.df["class"] == False]
            self.df_pos = self.df[self.df["class"] == True]
            n_pos = self.df_pos["ImageId"].nunique()
            neg_series = self.df_neg["ImageId"].unique()
            neg_series_selected = np.random.choice(
                neg_series, size=n_pos, replace=False
            )
            self.df_neg = self.df_neg[self.df_neg["ImageId"].isin(neg_series_selected)]
            self.df = pd.concat([self.df_pos, self.df_neg])

        # sample data
        if cfg.data.frac != 1 and split == "train":
            ids = self.df["ImageId"].unique()
            n_samples = int(len(ids) * cfg.data.frac)
            series_selected = np.random.choice(ids, size=n_samples, replace=False)
            self.df = self.df[self.df["ImageId"].isin(series_selected)]

        self.imgids = self.df.ImageId.unique().tolist()

        super(PneumothoraxImageDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, index):

        imgid = self.imgids[index]
        imgid_df = self.df.groupby("ImageId").get_group(imgid)

        # get image
        img_path = imgid_df.iloc[0]["Path"]
        x = self.read_from_dicom(img_path)

        # get labels
        if self.cfg.phase == "segmentation":
            rle_list = imgid_df[" EncodedPixels"].tolist()
            mask = np.zeros([1024, 1024])
            if rle_list[0] != " -1":
                for rle in rle_list:
                    mask += self.rle2mask(
                        rle, PNEUMOTHORAX_IMG_SIZE, PNEUMOTHORAX_IMG_SIZE
                    )
            mask = (mask >= 1).astype("float32")
            mask = self._resize_img(mask, self.cfg.data.image.imsize)

            augmented = self.seg_transform(image=x, mask=mask)
            x = augmented["image"]
            y = augmented["mask"].squeeze()
        else:
            y = imgid_df.iloc[0]["Label"]
            y = torch.tensor([y])

        return x, y

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array
        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        img = Image.fromarray(x).convert("RGB")
        return np.asarray(img)

    def __len__(self):
        return len(self.imgids)

    def rle2mask(self, rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position : current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T

    def get_transforms(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        list_transforms = []
        if self.split == "train":
            list_transforms.extend(
                [
                    ShiftScaleRotate(
                        shift_limit=0,  # no resizing
                        scale_limit=0.1,
                        rotate_limit=10,  # rotate
                        p=0.5,
                        border_mode=cv2.BORDER_CONSTANT,
                    )
                ]
            )
        list_transforms.extend(
            [
                Resize(self.cfg.data.image.imsize, self.cfg.data.image.imsize),
                Normalize(mean=mean, std=std, p=1),
                ToTensor(),
            ]
        )

        list_trfms = Compose(list_transforms)
        return list_trfms


class PneumoniaImageDataset(ImageBaseDataset):
    def __init__(self, cfg, split="train", transform=None):

        if PNEUMONIA_DATA_DIR is None:
            raise RuntimeError(
                "RNSA Pneumonia data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge"
                + f" and update PNEUMONIA_DATA_DIR in ./gloria/constants.py"
            )

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(PNEUMONIA_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(PNEUMONIA_VALID_CSV)
        else:
            self.df = pd.read_csv(PNEUMONIA_TEST_CSV)

        # only keep positive samples for detection
        if cfg.phase == "detection":
            self.df = self.df[self.df["Target"] == 1]

        # sample data
        if cfg.data.frac != 1 and split == "train":
            self.df = self.df.sample(frac=cfg.data.frac)

        super(PneumoniaImageDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # get image
        img_path = row["Path"]
        x = self.read_from_dicom(img_path)
        y = float(row["Target"])
        y = torch.tensor([y])
        return x, y

    def __len__(self):
        return len(self.df)

    def read_from_dicom(self, img_path):

        dcm = pydicom.read_file(img_path)
        x = dcm.pixel_array

        x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            x = cv2.bitwise_not(x)

        # transform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img
