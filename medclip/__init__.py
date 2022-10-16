name = 'MedCLIP'
version = '0.0.1'

from .modeling_medclip import (
    MedClipTextModel, # text encoder
    MedClipVisionModel, # vision encoder (ResNet50)
    MedClipVisionModelViT, # vision encoder (Swin-Transformer)
    MedClipModel, # vision-language encoders
    MedClipPromptClassifier, # make classification based manual prompts
    MedClipPromptTuningClassifier, # make classification based on prompt tuning
    MedClipClassifier, # make classification by appending a classifier to the vision encoder
)