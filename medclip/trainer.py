import os
import json
import pdb
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict
import math

import numpy as np
import torch
from torch import nn
from torch import device, Tensor
from tqdm.autonotebook import trange
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import distributed as dist
import transformers

WEIGHTS_NAME = "pytorch_model.bin"

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        eval_dataloader = None,
        evaluator=None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 100,
        save_steps : int = 100,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_total_limit: int = 0,
        load_best_model_at_last: bool = True,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.best_score = -9999999
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.score_logs = defaultdict(list)
        self.evaluator = evaluator
        self.eval_dataloader = eval_dataloader

        dataloaders = [dataloader for dataloader,_,_ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        loss_models = [loss for _, loss,_ in train_objectives]
        train_weights = [weight for _,_,weight in train_objectives]

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        # map models to devices
        model = model.cuda()

        # execute training on multiple GPUs
        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        train_loss_dict = defaultdict(list)
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            for train_iter in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):

                # check if model parameters keep same
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    loss_model.zero_grad()
                    loss_model.train()

                    loss_weight = train_weights[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        # for train_idx in range(num_train_objectives):
                        if '_build_prompt_sentence' in dir(dataloaders[train_idx].dataset):
                            dataloaders[train_idx].dataset._build_prompt_sentence()
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    if use_amp:
                        with autocast():
                            loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return['loss_value']
                        loss_value = loss_value
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return['loss_value'] / self.accumulation_steps
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    train_loss_dict[train_idx].append(loss_value.item())
                    optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps>0 and global_step % evaluation_steps == 0:
                    print('\n######### Train Loss #########')
                    for key in train_loss_dict.keys():
                        print('{} {:.4f} \n'.format(key, np.mean(train_loss_dict[key])))
                    train_loss_dict = defaultdict(list)

                    #TODO: update prompt sentences
                    # for train_idx in range(num_train_objectives):
                    #     if '_build_prompt_sentence' in dir(dataloaders[train_idx].dataset):
                    #         dataloaders[train_idx].dataset._build_prompt_sentence()

                if evaluation_steps > 0 and global_step % evaluation_steps == 0 and self.evaluator is not None:
                    scores = self.evaluator.evaluate()
                    print(f'\n######### Eval {global_step} #########')
                    for key in scores.keys():
                        if key in ['acc','auc']:
                            print('{}: {:.4f}'.format(key, scores[key]))
                    save_dir =  os.path.join(output_path, f'{global_step}/')
                    self._save_ckpt(model, save_dir)

                    # score logs save the list of scores
                    self.score_logs['global_step'].append(global_step)
                    for key in scores.keys():
                        if key in ['acc','auc']:
                            self.score_logs[key].append(scores[key])

                if self.evaluator is None and global_step % save_steps == 0:
                    state_dict = model.state_dict()
                    save_dir =  os.path.join(output_path, f'{global_step}/')
                    self._save_ckpt(model, save_dir)
                    print('model saved to', os.path.join(output_path, WEIGHTS_NAME))

        if save_best_model:
            import pandas as pd
            from distutils.dir_util import copy_tree
            res = pd.DataFrame(self.score_logs)
            res = res.set_index('global_step')
            # take the average column best
            best_iter = res.mean(1).idxmax()
            best_save_path = os.path.join(output_path, './best')
            if not os.path.exists(best_save_path): os.makedirs(best_save_path)
            best_origin_path = os.path.join(output_path, f'./{best_iter}')
            print(f'save best checkpoint at iter {best_iter} to', best_save_path)
            copy_tree(best_origin_path, best_save_path)

        if eval_dataloader is None and output_path is not None:   #No evaluator, but output path: save final model version
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_path, WEIGHTS_NAME))
            print('model saved to', os.path.join(output_path, WEIGHTS_NAME))

        if eval_dataloader is not None and load_best_model_at_last and save_best_model and evaluator is not None:
            state_dict = torch.load(os.path.join(best_save_path, WEIGHTS_NAME))
            model.load_state_dict(state_dict)
            print(f'load best checkpoint at last from {best_save_path}')

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, WEIGHTS_NAME))
