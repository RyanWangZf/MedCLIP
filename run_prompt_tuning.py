import os

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medclip import constants
from medclip.dataset import PromptTuningImageDataset, PromptTuningImageCollator
from medclip.evaluator import Evaluator
from medclip.modeling_medclip import MedClipModel, MedClipPromptTuningClassifier
from medclip.prompts import generate_class_prompts
from medclip.trainer import Trainer

# setup cuda devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# setup training configurations
# uncomment the following block for experiments
train_config = {
    'batch_size': 64,
    'num_epochs': 20,
    'warmup': 0.1, # the first 10% of training steps are used for warm-up
    'lr': 5e-4,
    'weight_decay': 0,
    'eval_batch_size': 256,
    'eval_steps': 50,
    'save_steps': 50,
    'n_context': 16,
    'class_specific_context': False,
}

# setup dataset name & config
# uncomment the following block for experiments

# ############################
# dataname = 'rsna'
# mode = 'binary'
# num_class = 2
# ############################

# ############################
dataname = 'chexpert-5x200'
# dataname = 'iuxray-5x200'
# dataname = 'mimic-5x200'
mode = 'multiclass'
num_class = 5
tasks = constants.CHEXPERT_COMPETITION_TASKS
# ############################

# ############################
# dataname = 'covid19-balance'
# mode = 'binary'
# tasks = constants.COVID_TASKS
# num_class = 2
# ############################

# generate class prompts by sampling from sentences
df_sent = pd.read_csv('./local_data/sentence-label.csv', index_col=0)

# build dataloader
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomAffine(degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)),
    transforms.Resize((256, 256)),
    transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])])

cls_prompts = generate_class_prompts(df_sent, task=constants.CHEXPERT_COMPETITION_TASKS, n=10)

train_data = PromptTuningImageDataset([f'{dataname}-train'],
                                      class_names=tasks,
                                      imgtransform=transform)
trainloader = DataLoader(train_data,
                         batch_size=train_config['batch_size'],
                         shuffle=True,
                         collate_fn=PromptTuningImageCollator(cls_prompts=cls_prompts,
                                                              mode=mode,
                                                              n_context=train_config['n_context'],
                                                              class_specific_context=train_config['class_specific_context']),
                         # num_workers=8,
                         )
val_data = PromptTuningImageDataset([f'{dataname}-test'],
                                    class_names=tasks)
valloader = DataLoader(val_data,
                       batch_size=train_config['eval_batch_size'],
                       shuffle=False,
                       collate_fn=PromptTuningImageCollator(cls_prompts=cls_prompts,
                                                            mode=mode,
                                                            n_context=train_config['n_context'],
                                                            class_specific_context=train_config['class_specific_context']),
                       # num_workers=4,
                       )

# load the pretrained model and build the classifier
model = MedClipModel(
    # checkpoint='./checkpoints/vision_text_pretrain/25000',
)
model.cuda()
if train_config['class_specific_context']:
    n_new_token = train_config['n_context'] * len(cls_prompts)
else:
    n_new_token = train_config['n_context']
clf = MedClipPromptTuningClassifier(model,
                                    ensemble=True,
                                    n_new_token=train_config['n_context'])
clf.cuda()
# for name, param in clf.named_parameters():
#     if name not in ['fc.weight', 'fc.bias']:
#         param.requires_grad = False

# build objective
train_objectives = [(trainloader, clf, 1)]
model_save_path = f'./checkpoints/{dataname}-prompttune'

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
