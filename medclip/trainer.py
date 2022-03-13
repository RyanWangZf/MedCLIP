import os
import json
import pdb
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch import device, Tensor
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers.util import import_from_string, batch_to_device, fullname, snapshot_download
from tqdm.autonotebook import trange
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import distributed as dist
import transformers

WEIGHTS_NAME = "pytorch_model.bin"

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return 

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

class TrainerDDP:
    '''trainer for multi-gpu training
    '''
    def __init__(self, args):
        # trigger multi-gpu
        print(torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
        print('world_size', torch.distributed.get_world_size()) # 打印当前进程数
        self.args = args
        torch.cuda.set_device(self.args.local_rank)

    def train(self,
        model,
        train_objectives: Iterable[Tuple[List, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        batch_size: int = 2,
        steps_per_epoch = None,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = transformers.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0
        ):
        self.best_score = -9999999
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        # build sampler and dataloader based on the train_objectives
        # datasets = [dataset for dataset,_ in train_objectives]
        # dataloaders = []
        # for dataset in datasets:
        #     dataloader = torch.utils.data.DataLoader(dataset,
        #                                  batch_size=batch_size,
        #                                  shuffle=True,
        #                                  num_workers=0,
        #                                  pin_memory=False,
        #                                  drop_last=True,
        #         )
        #     dataloaders.append(dataloader)
        # for dataloader in dataloaders:
        #     dataloader.collate_fn = model.smart_batching_collate
        
        dataloaders = [dataloader for dataloader,_ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)
        num_train_steps = num_train_steps // torch.distributed.get_world_size()

        loss_models = [loss for _, loss in train_objectives]
        
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
            scheduler_obj = model._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        # map models to devices
        model = model.cuda(self.args.local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                        device_ids=[self.args.local_rank], 
                                                        output_device=self.args.local_rank,
                                                        find_unused_parameters=False, 
                                                        broadcast_buffers=False)

        # execute training on multiple GPUs
        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)
        train_loss_dict = defaultdict(list)
        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        # loss_value = reduce_mean(loss_value, dist.get_world_size())
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    train_loss_dict[train_idx].append(loss_value.item())
                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if global_step % 100 == 0:
                    print('######### Train Loss #########')
                    for key in train_loss_dict.keys():
                        print('{} {:.4f} \n'.format(key, np.mean(train_loss_dict[key])))
                    train_loss_dict = defaultdict(list)

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    model.module._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    model.module._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            model.module._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
            model.module.save(output_path)

        if checkpoint_path is not None:
            model.module._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        eval_dataloader = None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupLinear',
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = transformers.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.best_score = -9999999
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()
        
        dataloaders = [dataloader for dataloader,_ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int(steps_per_epoch * epochs)
        loss_models = [loss for _, loss in train_objectives]
        self.eval_dataloader = eval_dataloader
        
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

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                # check if model parameters keep same
                pdb.set_trace()

                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(**data)
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(**data)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    train_loss_dict[train_idx].append(loss_value.item())
                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if global_step % 100 == 0:
                    print('######### Train Loss #########')
                    for key in train_loss_dict.keys():
                        print('{} {:.4f} \n'.format(key, np.mean(train_loss_dict[key])))
                    train_loss_dict = defaultdict(list)

                # if evaluation_steps > 0 and training_steps % evaluation_steps == 0 and self.eval_dataloader is not None:
                #     model._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)
                    # for loss_model in loss_models:
                    #     loss_model.zero_grad()
                    #     loss_model.train()

                # if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                    # model._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            # model._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if eval_dataloader is None and output_path is not None:   #No evaluator, but output path: save final model version
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_path, WEIGHTS_NAME))
            print('model saved to', os.path.joinn(output_path, WEIGHTS_NAME))        


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


