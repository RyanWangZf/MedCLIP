import numpy as np
import torch
import torch.nn.functional as F
import wandb
import json
import os
import segmentation_models_pytorch as smp

from pytorch_lightning.core import LightningModule
from .. import builder
from .. import gloria


class SegmentationModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

        if self.cfg.model.vision.model_name in gloria.available_models():
            self.model = gloria.load_img_segmentation_model(
                self.cfg.model.vision.model_name
            )
        else:
            self.model = smp.Unet("resnet50", encoder_weights=None, activation=None)

        self.loss = builder.build_loss(cfg)

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)

        if scheduler is None:
            return [optimizer]
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test", batch_idx)

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split, batch_idx):
        """Similar to traning step"""

        x, y = batch

        logit = self.model(x)
        logit = logit.squeeze()
        loss = self.loss(logit, y)

        prob = torch.sigmoid(logit)
        dice = self.get_dice(prob, y)
        # loss = self.loss(prob, y)

        if batch_idx == 0:
            img = batch[0][0].cpu().numpy()
            mask = batch[1][0].cpu().numpy()
            mask = np.stack([mask, mask, mask])

            layered = 0.6 * mask + 0.4 * img
            img = img.transpose((1, 2, 0))
            mask = mask.transpose((1, 2, 0))
            layered = layered.transpose((1, 2, 0))

            self.logger.experiment.log(
                {"input_image": [wandb.Image(img, caption="input_image")]}
            )
            self.logger.experiment.log({"mask": [wandb.Image(mask, caption="mask")]})
            self.logger.experiment.log(
                {"layered": [wandb.Image(layered, caption="layered")]}
            )
            self.logger.experiment.log({"pred": [wandb.Image(prob[0], caption="pred")]})

        # log_iter_loss = True if split == 'train' else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )

        return_dict = {"loss": loss, "dice": dice, "logit": logit, "prob": prob, "y": y}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        if self.cfg.lightning.trainer.distributed_backend == "dp":
            loss = (
                torch.stack([x["loss"] for x in step_outputs]).cpu().detach().tolist()
            )
        else:
            loss = [x["loss"].cpu().detach().item() for x in step_outputs]
        dice = [x["dice"] for x in step_outputs]
        loss = np.array(loss).mean()
        dice = np.array(dice).mean()

        self.log(f"{split}_dice", dice, on_epoch=True, logger=True, prog_bar=True)

        if split == "test":
            results_csv = os.path.join(self.cfg.output_dir, "results.csv")
            results = {"loss": loss, "dice": dice}
            with open(results_csv, "w") as fp:
                json.dump(results, fp)

    def get_dice(self, probability, truth, threshold=0.5):

        batch_size = len(truth)
        with torch.no_grad():
            probability = probability.view(batch_size, -1)
            truth = truth.view(batch_size, -1)
            assert probability.shape == truth.shape

            p = (probability > threshold).float()
            t = (truth > 0.5).float()

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = torch.nonzero(t_sum == 0)
            pos_index = torch.nonzero(t_sum >= 1)

            dice_neg = (p_sum == 0).float()
            dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = torch.cat([dice_pos, dice_neg])

        return torch.mean(dice).detach().item()
