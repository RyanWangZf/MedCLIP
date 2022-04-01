import pdb

import torch
from torch import nn

class ImageTextContrastiveLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    # contrastive loss function, adapted from
    # https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0
    
    def forward(self, 
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
        ):
        '''return loss values
        '''
        outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_loss=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                )
        
        return_res = {
            'loss_value': outputs.loss,
            'report': kwargs['report'],
            'image_embedding': outputs.image_embeds,
            'uid': kwargs['uid'],
        }
        return return_res

class ImageImageContrastiveLoss(nn.Module):
    '''compute contrastive loss using momentum memory banks between images and images,
    MOCO-V3 like.
    '''
    def __init__(self, model, momentum_model, T=1):
        '''
        T: temperature for infonce loss
        '''
        super().__init__()
        self.base_encoder = model
        self.momentum_encoder = momentum_model
        self.T = T
        # build momentum encoder
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
            param_m.to(param_b.device)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
        return loss

    def forward(self,
        input_ids=None,
        pixel_values=None,
        labels=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        '''
        * for normal/abnormal CL loss
            x1: normal images, x2: abnormal images
            q1: normal image feature, q2: abnormal image feature
            k1: normal features using momentum encoder
            k2: abnormal features using momentum encoder
            CL loss is applied for:
            negative pairs: {q1,k2}, {q2,k1}
            positive pairs: {q1,k1}, {q2,k2}
            to cl loss, it is {q1, k1, k2} and {q2, k2, k1}

        * for frontal/lateral CL loss:
            x1: frontal, x2: lateral
            q1, q2: encode x1, x2
            k1, k2: momentum encode x1, x2
            same as MOCO where {q1,k2} positive, {q2,k1} positive on diagonal
        '''
        feat_base = self.base_encoder.encode_image(pixel_values) # bs, 512
        feat_momtm = self.momentum_encoder.encode_image(pixel_values) # bs, 512

        if len(labels.unique()) == 1: # no abnormal images / normal images
            # only do instance discriminative loss
            loss = self.contrastive_loss(feat_base, feat_momtm)
        else:
            feat_base_nm = feat_base[labels==0]
            feat_base_ab = feat_base[labels==1]
            feat_momtm_nm = feat_momtm[labels==0]
            feat_momtm_ab = feat_momtm[labels==1]

            # q1,k1,k2
            loss_nm = self.contrastive_loss(
                feat_base_nm, 
                torch.cat([feat_momtm_nm, feat_momtm_ab], axis=0)
                )

            # q2,k2,k1
            loss_ab = self.contrastive_loss(
                feat_base_ab, 
                torch.cat([feat_momtm_ab, feat_momtm_nm], axis=0)
                )
            
            loss = loss_ab + loss_nm
        
        return_res = {
            'loss_value': loss,
            'uid': kwargs['uid'],
            'image_embedding': feat_base,
        }
        return return_res
