from data import Biview_Classification
from mlclassifier import MLClassifier
from my_build_vocab import Vocabulary
import os
import math
import argparse
import logging
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='dev')
    parser.add_argument('--model-path', type=str, default='./output/models')
    parser.add_argument('--pretrained', type=str, default='./weights/model_ones_3epoch_densenet.tar')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--dataset-dir', type=str, default='./data')
    parser.add_argument('--train-folds', type=str, default='012')
    parser.add_argument('--val-folds', type=str, default='3')
    parser.add_argument('--test-folds', type=str, default='4')
    parser.add_argument('--report-path', type=str, default='./data/reports.json')
    parser.add_argument('--vocab-path', type=str, default='./data/vocab.pkl')
    parser.add_argument('--label-path', type=str, default='./data/label_dict.json')
    parser.add_argument('--log-path', type=str, default='./output/logs')
    parser.add_argument('--log-freq', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--clip-value', type=float, default=5.0)
    parser.add_argument('--num-classes', type=int, default=20)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(args.log_path, args.name + '.log'), level=logging.INFO)
    print('------------------------Model and Training Details--------------------------')
    print(args)
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))

    writer = SummaryWriter(log_dir=os.path.join('./output/runs', args.name))

    device = torch.device('cuda:{}'.format(args.gpus[0]) if torch.cuda.is_available() else 'cpu')
    gpus = [int(_) for _ in list(args.gpus)]
    torch.manual_seed(args.seed)

    with open('./data/19class_keywords.txt') as f:
        keywords = f.read().splitlines()
    keywords.append('other')

    train_set = Biview_Classification('train', args.dataset_dir, args.train_folds, args.label_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_set = Biview_Classification('val', args.dataset_dir, args.val_folds, args.label_path)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
    test_set = Biview_Classification('test', args.dataset_dir, args.test_folds, args.label_path)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    model = MLClassifier(args.num_classes).to(device)
    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained, map_location=torch.device(device))
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        model_state_dict.update({k[7:]: v for k, v in pretrained_state_dict.items() if k[7:] in model_state_dict})
        model.load_state_dict(model_state_dict)

    BCELoss = nn.BCEWithLogitsLoss(pos_weight=train_set.get_class_weights()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150], gamma=0.1)

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    num_steps = math.ceil(len(train_set) / args.batch_size)

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        print('------------------------Training for Epoch {}---------------------------'.format(epoch))
        print('learning rate {:.7f}'.format(optimizer.param_groups[0]['lr']))
        for i, (images1, images2, labels) in enumerate(train_loader):
            images1, images2, labels = images1.to(device), images2.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images1, images2)
            loss = BCELoss(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= num_steps
        print('Epoch {}/{}, Loss {:.4f}'.format(epoch, args.num_epochs, epoch_loss))
        writer.add_scalar('train_loss', epoch_loss, epoch)

        scheduler.step(epoch)

        if epoch % args.log_freq == 0:

            save_fname = os.path.join(args.model_path, '{}_e{}.pth'.format(args.name, epoch))
            if len(gpus) > 1:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, save_fname)

            # evaluate
            model.eval()
            y = torch.zeros((len(val_set), args.num_classes), dtype=torch.int)
            y_score = torch.zeros((len(val_set), args.num_classes), dtype=torch.float)
            with torch.no_grad():
                for i, (images1, images2, labels) in enumerate(val_loader):
                    images1, images2 = images1.to(device), images2.to(device)
                    scores = torch.sigmoid(model(images1, images2)).detach().cpu()
                    y[i] = labels[0]
                    y_score[i] = scores[0]
            y_hat = (y_score >= 0.5).type(torch.int)
            y = y.numpy()
            y_score = y_score.numpy()
            y_hat = y_hat.numpy()
            precision, recall, f, _ = precision_recall_fscore_support(y, y_hat)
            roc_auc = roc_auc_score(y, y_score, average=None)
            print('Epoch {}/{}, P {:.4f}, R {:.4f}, F {:.4f}, AUC {:.4f}'.format(
                epoch, args.num_epochs, precision.mean(), recall.mean(), f.mean(), roc_auc.mean()))
            writer.add_scalar('precision', precision.mean(), epoch)
            writer.add_scalar('recall', recall.mean(), epoch)
            writer.add_scalar('f', f.mean(), epoch)
            writer.add_scalar('auc', roc_auc.mean(), epoch)

            # test
            y = torch.zeros((len(test_set), args.num_classes), dtype=torch.int)
            y_score = torch.zeros((len(test_set), args.num_classes), dtype=torch.float)
            with torch.no_grad():
                for i, (images1, images2, labels) in enumerate(test_loader):
                    images1, images2 = images1.to(device), images2.to(device)
                    scores = torch.sigmoid(model(images1, images2)).detach().cpu()
                    y[i] = labels[0]
                    y_score[i] = scores[0]
            y_hat = (y_score >= 0.5).type(torch.int)
            y = y.numpy()
            y_score = y_score.numpy()
            y_hat = y_hat.numpy()
            p, r, f, _ = precision_recall_fscore_support(y, y_hat)
            roc_auc = roc_auc_score(y, y_score, average=None)
            print('Epoch {}/{}, P {:.4f}, R {:.4f}, F {:.4f}, AUC {:.4f}'.format(
                epoch, args.num_epochs, p.mean(), r.mean(), f.mean(), roc_auc.mean()))
            df = np.stack([p, r, f, roc_auc], axis=1)
            df = pd.DataFrame(df, columns=['precision', 'recall', 'f1', 'auc'])
            df.insert(0, 'name', keywords)
            df.to_csv(os.path.join('./output/predictions', args.name + '_e{}.csv'.format(epoch)))


    writer.close()
