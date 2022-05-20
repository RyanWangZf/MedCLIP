from . import data_module
from . import image_dataset
from . import pretraining_dataset

DATA_MODULES = {
    "pretrain": data_module.PretrainingDataModule,
    "chexpert": data_module.CheXpertDataModule,
    "pneumothorax": data_module.PneumothoraxDataModule,
    "pneumonia": data_module.PneumoniaDataModule,
}
