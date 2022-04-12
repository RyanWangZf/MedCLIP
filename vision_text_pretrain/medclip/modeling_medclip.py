import pdb
import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np


from .vision_model import Uwinformer
from . import constants

class MedClipTextModel(nn.Module):
    def __init__(self, 
        bert_type=constants.BERT_TYPE,
        proj_dim = 512) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.projection_head = nn.Linear(768, proj_dim)
    
    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = torch.stack(output[2][-self.last_n_layer:]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3)
        embed = embed.mean(1)
        embed = self.projection_head(embed).mean(1)
        embed = embed / embed.norm(dim=-1, keepdim=True)
        return embed

class MedClipVisionModel(nn.Module):
    def __init__(self, checkpoint=None) -> None:
        super().__init__()
        self.vit_type = 'uwinformer'
        self.model = Uwinformer(
            img_size=256, 
            patch_size=4,
            in_chans=1, 
            proj_dim=512,
            embed_dim=128,
            num_heads=[4, 4, 4, 4],
            depths=[2,2,18,2],
            window_size=8,
            checkpoint=checkpoint,
        )

    def forward(self, pixel_values):
        '''args:
        pixel_values: tensor with shape [bs, 1, img_size, img_size]
        '''
        output = self.model(pixel_values)
        img_embeds = output / output.norm(dim=-1, keepdim=True)
        return img_embeds

class MedClipModel(nn.Module):
    def __init__(self,
        checkpoint=None,
        vision_checkpoint=None,
        logit_scale_init_value=0.07,
        ) -> None:
        super().__init__()
        self.vision_model = MedClipVisionModel(checkpoint=vision_checkpoint)
        self.text_model = MedClipTextModel()

        # learnable temperature for contrastive loss        
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))

        if checkpoint is not None:
            state_dict = torch.load(os.path.join(checkpoint, constants.WEIGHTS_NAME))
            self.load_state_dict(state_dict)
            print('load model weight from:', checkpoint)

    def forward(self, 
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        return_loss=None,
        **kwargs,
    ):
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        pixel_values = pixel_values.cuda()

        # image encoder
        vision_output = self.vision_model(pixel_values=pixel_values)
        img_embeds = vision_output / vision_output.norm(dim=-1, keepdim=True)

        # text encoder
        text_embeds = self.text_model(input_ids, attention_mask)

        # cosine similarity as logits, clamp logit scale for stability
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        
        # (n, d) dot (d, m) -> (n, m)
        logits_per_text = torch.matmul(text_embeds, img_embeds.t()) * logit_scale
        # (m, n): row 0: image to texts
        logits_per_image = logits_per_text.T

        if return_loss:
            loss = self.clip_loss(logits_per_text)
        else:
            loss = None
        
        return {'img_embeds':img_embeds, 'text_embeds':text_embeds,
            'logits':logits_per_image, 'loss_value':loss, 'logits_per_text':logits_per_text}

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

class MedClipPromptClassifier(nn.Module):
    '''take MedCLIP model with prompts for zero-shot classification
    '''
    def __init__(self, medclip_model) -> None:
        super().__init__()
        self.model = medclip_model

    def forward(self, pixel_values=None, prompt_inputs=None, **kwargs):
        '''take image pixel values (after transform) and prompt_inputs 
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pixel_values = pixel_values.cuda()
        class_similarities = []
        class_names = []
        for cls_name, cls_text in prompt_inputs.items():
            inputs = {'pixel_values':pixel_values}
            for k in cls_text.keys(): inputs[k] = cls_text[k].cuda()

            # TODO: 
            # take soft mask over class_prompts to reach the similarities to classes
            medclip_outputs = self.model(**inputs)
            logits = medclip_outputs['logits']

            # take logits max as the class similarity
            cls_sim = torch.max(logits, 1)[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)
        
        class_similarities = torch.stack(class_similarities, 1)
        outputs = {
            'logits': class_similarities,
            'class_names': class_names,
        }
        return outputs

