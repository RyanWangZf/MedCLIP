from torch import nn
import torch.nn.functional as F
import torch
import pdb

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
            
            logits = outputs['logits']

            # compute soft-labels
            # TODO: [zw] it seems similarity score of -1 is better than 0? However, the softmax is doing the opposite.
            label_sim = torch.matmul(img_labels, text_labels.T)
            label_sim = torch.clamp(label_sim, -1,1)
            label_sim = label_sim.to(logits.device)
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
        image_loss = self._soft_xent_loss(logits_per_img, F.softmax(soft_label,1))
        caption_loss = self._soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T,1))
        return (image_loss + caption_loss) / 2

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]