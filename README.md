# MedCLIP

Wang, Zifeng and Wu, Zhenbang and Agarwal, Dinesh and Sun, Jimeng. (2022). MedCLIP: Contrastive Learning from Unpaired Medical Images and Texts. EMNLP'22.


## Three lines to get pretrained MedCLIP models

```python
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel

# load MedCLIP-ResNet50
model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained()

# load MedCLIP-ViT
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
```
