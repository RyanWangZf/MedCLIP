import torch
import torch.nn as nn
import torchvision.transforms as transforms

from . import models
from . import lightning
from . import datasets
from . import loss


def build_data_module(cfg):
    if cfg.phase.lower() == "pretrain":
        data_module = datasets.DATA_MODULES["pretrain"]
    else:
        data_module = datasets.DATA_MODULES[cfg.data.dataset.lower()]
    return data_module(cfg)


def build_lightning_model(cfg, dm):
    module = lightning.LIGHTNING_MODULES[cfg.phase.lower()]
    module = module(cfg)
    module.dm = dm
    return module


def build_gloria_model(cfg):
    gloria_model = models.gloria_model.GLoRIA(cfg)
    return gloria_model


def build_gloria_from_ckpt(ckpt):

    ckpt = torch.load(ckpt)
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = build_gloria_model(cfg)
    gloria_model.load_state_dict(ckpt_dict)

    return gloria_model


def build_img_model(cfg):
    image_model = models.IMAGE_MODELS[cfg.phase.lower()]
    return image_model(cfg)


def build_text_model(cfg):
    return models.text_model.BertEncoder(cfg)


def build_optimizer(cfg, lr, model):

    # get params for optimization
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    # define optimizers
    if cfg.train.optimizer.name == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    elif cfg.train.optimizer.name == "Adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            betas=(0.5, 0.999),
        )
    elif cfg.train.optimizer.name == "AdamW":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=cfg.train.optimizer.weight_decay
        )


def build_scheduler(cfg, optimizer, dm=None):

    if cfg.train.scheduler.name == "warmup":

        def lambda_lr(epoch):
            if epoch <= 3:
                return 0.001 + epoch * 0.003
            if epoch >= 22:
                return 0.01 * (1 - epoch / 200.0) ** 0.9
            return 0.01

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    elif cfg.train.scheduler.name == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif cfg.train.scheduler.name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
    elif cfg.train.scheduler.name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    else:
        scheduler = None

    if cfg.lightning.trainer.val_check_interval is not None:
        cfg.train.scheduler.interval = "step"
        num_iter = len(dm.train_dataloader().dataset)
        if type(cfg.lightning.trainer.val_check_interval) == float:
            frequency = int(num_iter * cfg.lightning.trainer.val_check_interval)
            cfg.train.scheduler.frequency = frequency
        else:
            cfg.train.scheduler.frequency = cfg.lightning.trainer.val_check_interval

    scheduler = {
        "scheduler": scheduler,
        "monitor": cfg.train.scheduler.monitor,
        "interval": cfg.train.scheduler.interval,
        "frequency": cfg.train.scheduler.frequency,
    }

    return scheduler


def build_loss(cfg):

    if cfg.train.loss_fn.type == "DiceLoss":
        return loss.segmentation_loss.DiceLoss()
    elif cfg.train.loss_fn.type == "FocalLoss":
        return loss.segmentation_loss.FocalLoss()
    elif cfg.train.loss_fn.type == "MixedLoss":
        return loss.segmentation_loss.MixedLoss(alpha=cfg.train.loss_fn.alpha)
    elif cfg.train.loss_fn.type == "BCE":
        if cfg.train.loss_fn.class_weights is not None:
            weight = torch.Tensor(cfg.train.loss_fn.class_weights)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn
    else:
        raise NotImplementedError(f"{cfg.train.loss_fn} not implemented yet")


def build_transformation(cfg, split):

    t = []
    if split == "train":

        if cfg.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.transforms.random_crop.crop_size))

        if cfg.transforms.random_horizontal_flip is not None:
            t.append(
                transforms.RandomHorizontalFlip(p=cfg.transforms.random_horizontal_flip)
            )

        if cfg.transforms.random_affine is not None:
            t.append(
                transforms.RandomAffine(
                    cfg.transforms.random_affine.degrees,
                    translate=[*cfg.transforms.random_affine.translate],
                    scale=[*cfg.transforms.random_affine.scale],
                )
            )

        if cfg.transforms.color_jitter is not None:
            t.append(
                transforms.ColorJitter(
                    brightness=[*cfg.transforms.color_jitter.bightness],
                    contrast=[*cfg.transforms.color_jitter.contrast],
                )
            )
    else:
        if cfg.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            raise NotImplementedError("Normaliation method not implemented")

    return transforms.Compose(t)
