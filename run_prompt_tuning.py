import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants
from medclip.dataset import PromptTuningImageDataset, PromptTuningImageCollator
from medclip.evaluator import Evaluator
from medclip.modeling_medclip import MedClipModel, MedClipPromptTuningClassifier
from medclip.prompts import generate_class_prompts, generate_chexpert_class_prompts, generate_covid_class_prompts, \
    generate_rsna_class_prompts
from medclip.trainer import Trainer

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

# setup cuda devices
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# setup training configurations
train_config = {
    'batch_size': 64,
    'num_epochs': 10,
    'warmup': 0.1,  # the first 10% of training steps are used for warm-up
    'lr': 5e-4,
    'weight_decay': 0,
    'eval_batch_size': 256,
    'eval_steps': 50,
    'save_steps': 50,
    'n_context': 0,  # number of context tokens for prompt tuning
    'class_specific_context': False,  # if true, each class will have a different set of context tokens,
    'joint_train_emb': True,    # if true, the previous word embedings will be jointly trained
    'ensemble': True,
}

# uncomment the following block for experiments
dataname = 'chexpert-5x200'
# dataname = 'mimic-5x200'
# dataname = 'covid'
# dataname = 'rsna'

df_sent = pd.read_csv('./local_data/sentence-label.csv', index_col=0)
if dataname in ['chexpert-5x200', 'mimic-5x200']:
    tasks = constants.CHEXPERT_COMPETITION_TASKS
    num_class = 5
    mode = 'multiclass'
    train_dataname = f'{dataname}-finetune'
    val_dataname = dataname
    """ option 1: use prompts from sentence database """
    # cls_prompts = generate_class_prompts(df_sent, task=constants.CHEXPERT_COMPETITION_TASKS, n=10)
    """ option 2: use pre-defined prompts from constants.py """
    cls_prompts = generate_chexpert_class_prompts(n=10)
elif dataname == 'covid':
    tasks = constants.COVID_TASKS
    num_class = 2
    mode = 'binary'
    """ option 1: use entire training data """
    train_dataname = f'{dataname}-train'
    """ option 2: use 10% training data """
    # train_dataname = f'{dataname}-0.1-train'
    val_dataname = f'{dataname}-test'
    cls_prompts = generate_class_prompts(df_sent, ['No Finding'], n=10)
    covid_prompts = generate_covid_class_prompts(n=10)
    cls_prompts.update(covid_prompts)
elif dataname == 'rsna':
    tasks = constants.RSNA_TASKS
    num_class = 2
    mode = 'binary'
    train_dataname = f'{dataname}-train'
    val_dataname = f'{dataname}-test'
    cls_prompts = generate_class_prompts(df_sent, ['No Finding'], n=10)
    rsna_prompts = generate_rsna_class_prompts(n=10)
    cls_prompts.update(rsna_prompts)
else:
    raise NotImplementedError

""" option: use class name as prompts """
# cls_prompts = {task: task for task in tasks}

# build dataloader
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])])

train_data = PromptTuningImageDataset([train_dataname],
                                      class_names=tasks,
                                      imgtransform=transform)
collate_fn = PromptTuningImageCollator(cls_prompts=cls_prompts,
                                       mode=mode,
                                       n_context=train_config['n_context'],
                                       class_specific_context=train_config['class_specific_context'])
trainloader = DataLoader(train_data,
                         batch_size=train_config['batch_size'],
                         shuffle=True,
                         collate_fn=collate_fn,
                         num_workers=8,
                         )
val_data = PromptTuningImageDataset([val_dataname],
                                    class_names=tasks)
valloader = DataLoader(val_data,
                       batch_size=train_config['eval_batch_size'],
                       shuffle=False,
                       collate_fn=collate_fn,
                       num_workers=4,
                       )

# load the pretrained model and build the classifier
model = MedClipModel(
    checkpoint='/srv/local/data/MedCLIP/checkpoints/vision_text_pretrain/21000/'
)
model.cuda()
clf = MedClipPromptTuningClassifier(model,
                                    ensemble=train_config['ensemble'],
                                    n_context=train_config['n_context'],
                                    class_specific_context=train_config['class_specific_context'],
                                    num_class=num_class,
                                    mode=mode,
                                    joint_train_emb=train_config['joint_train_emb'])
clf.cuda()
for name, param in clf.named_parameters():
    if 'text_model.model.embeddings.word_embeddings' not in name:
        param.requires_grad = False

# build objective
train_objectives = [(trainloader, clf, 1)]
model_save_path = f'./checkpoints/{dataname}-prompt-tuning'

# build trainer
trainer = Trainer()

evaluator = Evaluator(
    medclip_clf=clf,
    eval_dataloader=valloader,
    mode=mode,
)
trainer.train(
    clf,
    train_objectives=train_objectives,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr': train_config['lr']},
    output_path=model_save_path,
    evaluation_steps=train_config['eval_steps'],
    weight_decay=train_config['weight_decay'],
    save_steps=train_config['save_steps'],
    evaluator=evaluator,
    eval_dataloader=valloader,
    use_amp=False,
)

# complete training and start to evaluate
res = evaluator.evaluate()
for key in res.keys():
    if key not in ['pred', 'labels']:
        print(f'{key}: {res[key]}')
