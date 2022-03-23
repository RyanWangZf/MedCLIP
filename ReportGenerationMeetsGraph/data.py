import os
import random
import json
import pickle
import string
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Biview_Classification(Dataset):

    def __init__(self, phase, dataset_dir, folds, label_fname):
        self.phase = phase
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        self.transform = transforms.Compose([
            transforms.RandomCrop((512, 512), pad_if_needed=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        img1_path = os.path.join('./data/NLMCXR_pngs', img1_path)
        img2_path = os.path.join('./data/NLMCXR_pngs', img2_path)
        image1 = Image.open(img1_path)
        image1 = self.transform(image1)
        image2 = Image.open(img2_path)
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        return image1, image2, label

    def get_class_weights(self):
        all_labels = [v for k, v in self.label_dict.items()]
        all_labels = torch.tensor(all_labels, dtype=torch.int)
        num_cases, num_classes = all_labels.size()
        pos_counts = torch.sum(all_labels, dim=0)
        neg_counts = num_cases - pos_counts
        ratio = neg_counts.type(torch.float) / pos_counts.type(torch.float)
        return ratio


class Biview_OneSent(Dataset):

    def __init__(self, phase, dataset_dir, folds, report_fname, vocab_fname, label_fname):
        self.phase = phase
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        with open(vocab_fname, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        self.transform = transforms.Compose([
            transforms.RandomCrop((512, 512), pad_if_needed=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(img1_path)
        image1 = self.transform(image1)
        image2 = Image.open(img2_path)
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        report = self.reports[caseid]
        text = ''
        if report['impression'] is not None:
            text += report['impression']
        text += ' '
        if report['findings'] is not None:
            text += report['findings']
        text = text.lower().translate(str.maketrans('', '', string.punctuation.replace('.', '')))
        text = text.replace('.', ' .')
        tokens = text.strip().split()
        caption = [self.vocab('<start>'), *[self.vocab(token) for token in tokens], self.vocab('<end>')]
        if len(caption) == 2:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        caption = torch.tensor(caption)

        return image1, image2, label, caption, caseid


def collate_fn(data):
    # data.sort(key=lambda x: x[-1], reverse=True)
    images1, images2, labels, captions, caseids = zip(*data)

    images1 = torch.stack(images1, 0)
    images2 = torch.stack(images2, 0)
    labels = torch.stack(labels, 0)

    max_len = max([len(cap) for cap in captions])
    targets = torch.zeros((len(captions), max_len), dtype=torch.long)
    masks = torch.zeros((len(captions), max_len), dtype=torch.uint8)
    for icap, cap in enumerate(captions):
        l = len(cap)
        targets[icap, :l] = cap
        masks[icap, :l].fill_(1)

    return images1, images2, labels, targets, masks, caseids


class Biview_MultiSent(Dataset):

    def __init__(self, phase, dataset_dir, folds, report_fname, vocab_fname, label_fname):
        self.phase = phase
        self.case_list = []
        for fold in list(folds):
            with open(os.path.join(dataset_dir, 'fold{}.txt'.format(fold))) as f:
                self.case_list += f.read().splitlines()
        with open(report_fname) as f:
            self.reports = json.load(f)
        with open(vocab_fname, 'rb') as f:
            self.vocab = pickle.load(f)
        with open(label_fname) as f:
            self.label_dict = json.load(f)
        self.transform = transforms.Compose([
            transforms.RandomCrop((512, 512), pad_if_needed=True),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        caseid, img1_path, img2_path = self.case_list[idx].split()
        image1 = Image.open(img1_path)
        image1 = self.transform(image1)
        image2 = Image.open(img2_path)
        image2 = self.transform(image2)

        label = self.label_dict[caseid]
        label = torch.tensor(label, dtype=torch.float)

        report = self.reports[caseid]
        text = ''
        if report['impression'] is not None:
            text += report['impression']
        text += ' '
        if report['findings'] is not None:
            text += report['findings']
        sents = text.lower().split('.')
        sents = [sent for sent in sents if len(sent.strip()) > 1]
        caption = []
        for isent, sent in enumerate(sents):
            tokens = sent.translate(str.maketrans('', '', string.punctuation)).strip().split()
            caption.append([self.vocab('.'), *[self.vocab(token) for token in tokens], self.vocab('.')])
        if caption == []:
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        caption[0][0] = self.vocab('<start>')
        caption[-1].append(self.vocab('<end>'))

        return image1, image2, label, caption, caseid


def sent_collate_fn(data):
    # data.sort(key=lambda x: x[-1], reverse=True)
    images1, images2, labels, captions, caseids = zip(*data)

    images1 = torch.stack(images1, 0)
    images2 = torch.stack(images2, 0)
    labels = torch.stack(labels, 0)

    num_sents = [len(cap) for cap in captions]
    sent_lens = [len(sent) for cap in captions for sent in cap]
    max_num_sents = max(num_sents) if len(num_sents) > 0 else 1
    max_sent_lens = max(sent_lens) if len(sent_lens) > 0 else 1
    targets = torch.zeros((len(captions), max_num_sents, max_sent_lens), dtype=torch.long)
    loss_masks = torch.zeros((len(captions), max_num_sents, max_sent_lens), dtype=torch.uint8)
    update_masks = torch.zeros((len(captions), max_num_sents, max_sent_lens), dtype=torch.uint8)
    for icap, cap in enumerate(captions):
        for isent, sent in enumerate(cap):
            l = len(sent)
            assert (l > 0)
            targets[icap, isent, :l] = torch.tensor(sent, dtype=torch.long)
            loss_masks[icap, isent, 1:l].fill_(1)
            update_masks[icap, isent, :l-1].fill_(1)

    return images1, images2, labels, targets, loss_masks, update_masks, caseids
