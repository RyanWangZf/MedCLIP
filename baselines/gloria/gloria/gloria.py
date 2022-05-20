import os
import torch
import numpy as np
import copy
import random
import pandas as pd
import segmentation_models_pytorch as smp
import pdb

from . import builder
from . import utils
from . import constants
from .models.vision_model import PretrainedImageClassifier
from typing import Union, List


np.random.seed(6)
random.seed(6)


_MODELS = {
    "gloria_resnet50": "./pretrained/chexpert_resnet50.ckpt",
    "gloria_resnet18": "./pretrained/chexpert_resnet18.ckpt",
}


_SEGMENTATION_MODELS = {
    "gloria_resnet50": "./pretrained/chexpert_resnet50.ckpt",
}


_FEATURE_DIM = {"gloria_resnet50": 2048, "gloria_resnet18": 2048}


def available_models() -> List[str]:
    """Returns the names of available GLoRIA models"""
    return list(_MODELS.keys())


def available_segmentation_models() -> List[str]:
    """Returns the names of available GLoRIA models"""
    return list(_SEGMENTATION_MODELS.keys())


def load_gloria(
    name: str = "gloria_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a GLoRIA model

    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    gloria_model : torch.nn.Module
        The GLoRIA model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            + "Make sure to download the pretrained weights from \n"
            + "    https://stanfordmedicine.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh \n"
            + " and copy it to the ./pretrained folder."
        )

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    # DEBUG: avoid error raised due to long input texts
    cfg['data']['text']['word_num'] = 512

    gloria_model = builder.build_gloria_model(cfg).to(device)
    gloria_model.load_state_dict(ckpt_dict)

    return gloria_model


def load_img_classification_model(
    name: str = "gloria_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    num_cls: int = 1,
    freeze_encoder: bool = True,
):
    """Load a GLoRIA pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    num_cls: int
        Number of output classes
    freeze_encoder: bool
        Freeze the pretrained image encoder

    Returns
    -------
    img_model : torch.nn.Module
        The GLoRIA pretrained image classification model
    """

    # load pretrained image encoder
    gloria_model = load_gloria(name, device)
    image_encoder = copy.deepcopy(gloria_model.img_encoder)
    del gloria_model

    # create image classifier
    feature_dim = _FEATURE_DIM[name]
    img_model = PretrainedImageClassifier(
        image_encoder, num_cls, feature_dim, freeze_encoder
    )

    return img_model


def load_img_segmentation_model(
    name: str = "gloria_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a GLoRIA pretrained classification model

    Parameters
    ----------
    name : str
        A model name listed by `gloria.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    img_model : torch.nn.Module
        The GLoRIA pretrained image classification model
    """

    # warnings
    if name in _SEGMENTATION_MODELS:
        ckpt_path = _SEGMENTATION_MODELS[name]
        base_model = name.split("_")[-1]
    elif os.path.isfile(name):
        ckpt_path = name
        base_model = "resnet50"  # TODO
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_segmentation_models()}"
        )

    # load base model
    seg_model = smp.Unet(base_model, encoder_weights=None, activation=None)

    # update weight
    ckpt = torch.load(ckpt_path)
    ckpt_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("gloria.img_encoder.model"):
            k = ".".join(k.split(".")[3:])
            ckpt_dict[k] = v
        ckpt_dict["fc.bias"] = None
        ckpt_dict["fc.weight"] = None
    seg_model.encoder.load_state_dict(ckpt_dict)

    return seg_model.to(device)


def get_similarities(gloria_model, imgs, txts, similarity_type="both"):
    """Load a GLoRIA pretrained classification model

    Parameters
    ----------
    gloria_model : str
        GLoRIA model, load via gloria.load_models()
    imgs:
        processed images using gloria_model.process_img
    txts:
        processed text using gloria_model.process_text
    similartiy_type
        Either local, global or both

    Returns
    -------
    similarities :
        similartitie between each imgs and text
    """

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use gloria_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use gloria_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        img_emb_l, img_emb_g = gloria_model.image_encoder_forward(imgs)
        text_emb_l, text_emb_g, _ = gloria_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

    # get similarities
    global_similarities = gloria_model.get_global_similarities(img_emb_g, text_emb_g)
    local_similarities = gloria_model.get_local_similarities(
        img_emb_l, text_emb_l, txts["cap_lens"]
    )
    similarities = (local_similarities + global_similarities) / 2

    if similarity_type == "global":
        return global_similarities.detach().cpu().numpy()
    elif similarity_type == "local":
        return local_similarities.detach().cpu().numpy()
    else:
        return similarities.detach().cpu().numpy()


def zero_shot_classification(gloria_model, imgs, cls_txt_mapping):
    """Load a GLoRIA pretrained classification model

    Parameters
    ----------
    gloria_model : str
        GLoRIA model, load via gloria.load_models()
    imgs:
        processed images using gloria_model.process_img
    cls_txt_mapping:
        dictionary of class to processed text mapping. Each class can have more than one associated text

    Returns
    -------
    cls_similarities :
        similartitie between each imgs and text
    """

    # get similarities for each class
    class_similarities = []
    for cls_name, cls_txt in cls_txt_mapping.items():
        similarities = get_similarities(
            gloria_model, imgs, cls_txt, similarity_type="both"
        )
        cls_similarity = similarities.max(axis=1)  # average between class prompts
        class_similarities.append(cls_similarity)
    class_similarities = np.stack(class_similarities, axis=1)

    # normalize across class
    if class_similarities.shape[0] > 1:
        class_similarities = utils.normalize(class_similarities)
    class_similarities = pd.DataFrame(
        class_similarities, columns=cls_txt_mapping.keys()
    )

    return class_similarities


def generate_chexpert_class_prompts(n: int = 5):
    """Generate text prompts for each CheXpert classification task

    Parameters
    ----------
    n:  int
        number of prompts per class

    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in constants.CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        prompts[k] = random.sample(cls_prompts, n)
    return prompts
