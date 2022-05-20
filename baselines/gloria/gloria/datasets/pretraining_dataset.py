import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data

from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from gloria.constants import *


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, cfg, split="train", transform=None):

        if CHEXPERT_DATA_DIR is None:
            raise RuntimeError(
                "CheXpert data path empty\n"
                + "Make sure to download data from:\n"
                + "    https://stanfordmlgroup.github.io/competitions/chexpert/"
                + f" and update CHEXPERT_DATA_DIR in ./gloria/constants.py"
            )

        self.cfg = cfg
        self.transform = transform
        self.max_word_num = self.cfg.data.text.captions_per_image

        # read CheXpert csv file
        csv_path = os.path.join(CHEXPERT_DATA_DIR, CHEXPERT_MASTER_CSV)
        self.df = pd.read_csv(csv_path)
        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(
            lambda x: os.path.join(CHEXPERT_DATA_DIR, "/".join(x.split("/")[1:]))
        )
        self.df = self.df[self.df[CHEXPERT_VIEW_COL] == "Frontal"]

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def load_text_data(self, split):

        # get study to captions mapping
        filepath = os.path.join(CHEXPERT_DATA_DIR, "captions.pickle")
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, to_remove = self.create_path_2_sent_mapping(
                self.df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump([path2sent, to_remove], f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent, to_remove = pickle.load(f)

        # filter studies to use for current split
        filenames = self.df[self.df[CHEXPERT_SPLIT_COL] == split][
            CHEXPERT_PATH_COL
        ].tolist()
        filenames = [f for f in filenames if f not in to_remove]

        return filenames, path2sent

    def get_caption(self, path):

        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            print(path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(series_sents)
        else:
            sent_ix = random.randint(0, len(series_sents))
            sent = series_sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, transform=None):

        x = cv2.imread(str(img_path), 0)

        # tranform images
        x = self._resize_img(x, self.cfg.data.image.imsize)
        img = Image.fromarray(x).convert("RGB")

        if transform is not None:
            img = transform(img)

        return img

    def __getitem__(self, index):

        key = self.filenames[index]

        imgs = self.get_imgs(key, self.transform)

        # randomly select a sentence
        caps, cap_len = self.get_caption(key)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)

    def create_path_2_sent_mapping(self, df, max_word_num):

        sent_lens, num_sents, to_remove = [], [], []
        path2sent = {}
        for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):

            # pick impression, findings, last_paragraph
            captions = ""
            if type(row[CHEXPERT_REPORT_COL]) == str:
                captions += row[CHEXPERT_REPORT_COL]

            # remove empty reports
            if len(captions) == 0:
                to_remove.append(row[CHEXPERT_PATH_COL])

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:

                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    # if len(tokens) < 3:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                study_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                cnt += len(included_tokens)
                if cnt == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(study_sent))

            # remove paths without setnences
            if len(study_sent) > 0:
                path2sent[row[CHEXPERT_PATH_COL]] = study_sent
            else:
                to_remove.append(row[CHEXPERT_PATH_COL])

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent, to_remove

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


def multimodal_collate_fn(batch):
    """sort sequence"""

    imgs, cap_len, ids, tokens, attention, path = [], [], [], [], [], []

    # flattern
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path,
    }

    return return_dict
