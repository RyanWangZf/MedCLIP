from .pretrain_model import PretrainModel
from .classification_model import ClassificationModel
from .segmentation_model import SegmentationModel

LIGHTNING_MODULES = {
    "pretrain": PretrainModel,
    "classification": ClassificationModel,
    "segmentation": SegmentationModel,
}
