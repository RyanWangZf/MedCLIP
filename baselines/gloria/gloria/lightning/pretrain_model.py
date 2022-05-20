import torch

from PIL import Image
from .. import builder
from .. import loss
from .. import utils

from pytorch_lightning.core import LightningModule
from torch.autograd import Variable


class PretrainModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters(self.cfg)
        self.gloria = builder.build_gloria_model(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.gloria)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss, attn_maps, sents = self.shared_step(batch, "train")

        # get attention map image
        if self.cfg.train.update_interval is not None:
            if batch_idx % self.cfg.train.update_interval == 0:
                imgs = batch["imgs"].cpu()
                self.gloria.plot_attn_maps(
                    attn_maps, imgs, sents, self.current_epoch, batch_idx
                )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, "val")
        return loss

    def shared_step(self, batch, split):
        """Similar to traning step"""

        img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents = self.gloria(batch)
        loss, attn_maps = self.gloria.calc_loss(
            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents
        )

        # log training progress
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return loss, attn_maps, sents
