from torch import nn
import torch.nn.functional as F
import torch
import pdb
import numpy as np

class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        img_labels=None,
        text_labels=None,
        aug_input_ids=None,
        aug_attention_mask=None,
        **kwargs,
        ):
        '''args:
        labels: the image corresponds to which classes of diagnoses
        text_labels: the text corresponds to which classes of diagnoses
        '''
        if img_labels is None or text_labels is None:
            '''use hard clip loss as the original clip
            '''
            outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=True,
                    )
        else:
            '''use soft clip loss
            '''
            outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    return_loss=False,
                    )

            # get logits
            logits = outputs['logits']

            # compute soft-labels, -1: negative, 0: uncertain, 1: positive
            # in the original data: 1: positive, 0: negative, -1: uncertain, NA: not mentioned
            label_sim = torch.matmul(img_labels, text_labels.T)
            label_sim = label_sim.to(logits.device)

            if aug_input_ids is not None:
                aug_text_embeds = self.model.encode_text(aug_input_ids, aug_attention_mask)
                img_embeds = outputs['img_embeds']
                logits_aug = self.model.compute_logits(img_embeds, aug_text_embeds)
                aug_loss_value = self._soft_clip_loss(logits_aug, label_sim)
                loss_value = self._soft_clip_loss(logits, label_sim)
                outputs['loss_value'] = (aug_loss_value + loss_value) / 2
            else:
                outputs['loss_value'] = self._soft_clip_loss(logits, label_sim)

        return_res = {
            'loss_value': outputs['loss_value'],
        }
        return return_res

    def _soft_clip_loss(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        image_loss = self._soft_xent_loss(logits_per_img, F.softmax(soft_label,1))
        caption_loss = self._soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T,1))
        return (image_loss + caption_loss) / 2

        # when using multilabel bce loss
        # image_loss = self._soft_bce_loss(logits_per_img, soft_label)
        # return image_loss

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]

    def _soft_bce_loss(self, input, target):
        return nn.functional.binary_cross_entropy_with_logits(input, target)


class ImageSuperviseLoss(nn.Module):
    def __init__(self,
        model,
        loss_fn=None,
        ):
        super().__init__()
        self.model = model
        self.mode = model.mode
        if loss_fn is None:
            if self.mode in ['multilabel','binary']:
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

    def forward(self,
        pixel_values,
        labels=None,
        **kwargs):
        outputs = self.model(pixel_values=pixel_values, labels=labels, return_loss=True)
        # mix_x, y_a, y_b, lamb = self.mixup_data(pixel_values, labels)
        # outputs = self.model(pixel_values=mix_x, labels=labels, return_loss=False)
        # y_a = y_a.cuda()
        # y_b = y_b.cuda()
        # loss = self.mixup_criterion(self.loss_fn, outputs['logits'], y_a, y_b, lamb)
        # outputs['loss_value'] = loss
        return outputs

    def mixup_data(self, x, y, alpha=0.3):
        if alpha > 0: lamb = np.random.beta(alpha, alpha)
        else: lamb = 1
        batch_size = x.shape[0]
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lamb * x + (1 - lamb) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lamb

    def mixup_criterion(self, criterion, pred, y_a, y_b, lamb):
        return lamb * criterion(pred, y_a) + (1- lamb) * criterion(pred, y_b)
