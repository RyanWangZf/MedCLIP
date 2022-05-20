from pathlib import Path

# CheXpert constants
CHEXPERT_DATA_DIR = Path("/home/mars/git/embeddingx_v2/data/chexpert/CheXpert-v1.0")
CHEXPERT_ORIGINAL_TRAIN_CSV = CHEXPERT_DATA_DIR / "train.csv"
CHEXPERT_TRAIN_CSV = CHEXPERT_DATA_DIR / "train_split.csv"  # train split from train.csv
CHEXPERT_VALID_CSV = CHEXPERT_DATA_DIR / "valid_split.csv"  # valid split from train.csv
CHEXPERT_TEST_CSV = (
    CHEXPERT_DATA_DIR / "valid.csv"
)  # using validation set as test set (test set label hidden)
CHEXPERT_MASTER_CSV = (
    CHEXPERT_DATA_DIR / "master_updated.csv"
)  # contains patient information, not PHI conplient
CHEXPERT_TRAIN_DIR = CHEXPERT_DATA_DIR / "train"
CHEXPERT_TEST_DIR = CHEXPERT_DATA_DIR / "valid"
CHEXPERT_5x200 = CHEXPERT_DATA_DIR / "chexpert_8x200.csv"

CHEXPERT_VALID_NUM = 5000
CHEXPERT_VIEW_COL = "Frontal/Lateral"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_SPLIT_COL = "Split"
CHEXPERT_REPORT_COL = "Report Impression"

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

# baseed on original chexpert paper
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}

# SIIM Pneumothorax
PNEUMOTHORAX_DATA_DIR = Path("/data4/siim")
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR / "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR / "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7

# RSNA Pneumonia
PNEUMONIA_DATA_DIR = Path("/data4/rsna_pneumonia/")  # aimi0/rsna_pneumonia')
PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_DATA_DIR / "stage_2_train_labels.csv"
PNEUMONIA_TRAIN_CSV = PNEUMONIA_DATA_DIR / "train.csv"
PNEUMONIA_VALID_CSV = PNEUMONIA_DATA_DIR / "val.csv"
PNEUMONIA_TEST_CSV = PNEUMONIA_DATA_DIR / "test.csv"
PNEUMONIA_IMG_DIR = PNEUMONIA_DATA_DIR / "stage_2_train_images"
PNEUMONIA_TRAIN_PCT = 0.7


CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}
