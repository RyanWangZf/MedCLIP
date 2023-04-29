# MedCLIP

[![PyPI version](https://badge.fury.io/py/medclip.svg)](https://badge.fury.io/py/medclip)
[![Downloads](https://pepy.tech/badge/medclip)](https://pepy.tech/project/medclip)
![GitHub Repo stars](https://img.shields.io/github/stars/ryanwangzf/medclip)
![GitHub Repo forks](https://img.shields.io/github/forks/ryanwangzf/medclip)


Wang, Zifeng and Wu, Zhenbang and Agarwal, Dinesh and Sun, Jimeng. (2022). MedCLIP: Contrastive Learning from Unpaired Medical Images and Texts. EMNLP'22.

[Paper PDF](https://arxiv.org/pdf/2210.10163.pdf)

## Download MedCLIP
Before download MedCLIP, you need to find feasible torch version (with GPU) on https://pytorch.org/get-started/locally/.

Then, download MedCLIP by

```bash
pip install git+https://github.com/RyanWangZf/MedCLIP.git

# or

pip install medclip
```

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

## As simple as using CLIP

```python
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip.processing import CLIPTokenizer
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


# instantiate CLIPTokenizer object
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

# instantiate MedCLIPProcessor object
processor = MedCLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', image_size=224)

# load demo image and apply transforms
image = Image.open('/content/drive/MyDrive/RadiXClean/content/all_images/PMC107839_1471-2296-3-6-2.jpg').convert("RGB")
image_transforms = Compose([
    Resize(256, interpolation=3),
    CenterCrop(224),
    ToTensor()
])
image = image_transforms(image)
image = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                  std=[0.26862954, 0.26130258, 0.27577711])(image)

# prepare input for MedCLIP model
inputs = processor(
    text=["lungs remain severely hyperinflated with upper lobe emphysema", 
          "opacity left costophrenic angle is new since prior exam ___ represent some loculated fluid cavitation unlikely"], 
    images=image.unsqueeze(0), 
    padding=True,
    return_tensors="pt"
)

# load pre-trained MedCLIP model and perform inference
model = MedCLIPModel.from_pretrained('openai/clip-vit-base-patch32', tokenizer=tokenizer)
model.cuda()
outputs = model(**inputs)

print(outputs.keys())
# dict_keys(['img_embeds', 'text_embeds', 'logits', 'loss_value', 'logits_per_text'])

```

## MedCLIP for Prompt-based Classification

```python
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier

processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
clf = PromptClassifier(model, ensemble=True)
clf.cuda()

# prepare input image
from PIL import Image
image = Image.open('./example_data/view1_frontal.jpg')
inputs = processor(images=image, return_tensors="pt")

# prepare input prompt texts
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
inputs['prompt_inputs'] = cls_prompts

# make classification
output = clf(**inputs)
print(output)
# {'logits': tensor([[0.5154, 0.4119, 0.2831, 0.2441, 0.4588]], device='cuda:0',
#       grad_fn=<StackBackward0>), 'class_names': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']}
```

## How to Get Sentence-level Semantic Labels

You can refer to https://github.com/stanfordmlgroup/chexpert-labeler where wonderful information extraction tools are offered!
