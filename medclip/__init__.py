import numpy as np

# Patch lỗi tương thích Numpy >= 1.24
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'object'): np.object = object
name = 'MedCLIP'
version = '0.0.3'

from .modeling_medclip import (
    MedCLIPTextModel, # text encoder
    MedCLIPVisionModel, # vision encoder (ResNet50)
    MedCLIPVisionModelViT, # vision encoder (Swin-Transformer)
    MedCLIPModel, # vision-language encoders
    PromptClassifier, # make classification based manual prompts
    PromptTuningClassifier, # make classification based on prompt tuning
    SuperviseClassifier, # make classification by appending a classifier to the vision encoder
)

from .dataset import MedCLIPProcessor