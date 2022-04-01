import pdb
import os

from transformers import CLIPProcessor, CLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput
import torch
from torch import nn

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0

class MedCLIPModel(nn.Module):
    def __init__(self, load_dir=None):
        super().__init__()
        if load_dir is not None:
            if not os.path.exists(load_dir):
                print(f'no pretrained model found in {load_dir}, load from Huggingface')
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            else:
                print(f'load pretrained MedClip from local dir {load_dir}')
                self.clip = CLIPModel.from_pretrained(load_dir)
        else:
            print('initialize medclip with CLIP from huggingface')
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        self.config = self.clip.config
        # define the visual parts
        self.pixel_input_conv = nn.Conv2d(1,3,1,bias=False)
        self.vision_model = nn.Sequential(
            self.pixel_input_conv,
            self.clip.vision_model,
        )
        self.visual_projection = self.clip.visual_projection
        
        # store a mapping between uid to image embedding
        self.image_embedding_bank = dict()

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # make input pixel values gray scale to RGB three channels with convolution
        pixel_values = pixel_values.to(self.clip.device).float()
        pixel_values = self.pixel_input_conv(pixel_values.unsqueeze(1))
        vision_outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        image_embeds = vision_outputs[1]
        image_embeds = self.clip.visual_projection(image_embeds)

        input_ids = input_ids.to(self.clip.device)
        attention_mask = attention_mask.to(self.clip.device)
        text_outputs = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        text_embeds = text_outputs[1]
        text_embeds = self.clip.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def encode_image(self, pixel_values=None, normalize=True):
        '''receive pixel values, extract the image features
        '''
        pixel_values = pixel_values.to(self.clip.device).float()
        vision_outputs = self.vision_model(pixel_values.unsqueeze(1))
        image_embeds = vision_outputs[1]
        image_embeds = self.clip.visual_projection(image_embeds)
        if normalize: image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds
    
    def encode_text(self, input_ids=None, attention_mask=None, normalize=True):
        '''receive tokenized texts, extract the text features
        '''
        input_ids = input_ids.to(self.clip.device)
        attention_mask = attention_mask.to(self.clip.device)
        text_embeds = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)['pooler_output']
        text_embeds = self.clip.text_projection(text_embeds)
        if normalize: text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds