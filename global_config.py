from pathlib import Path
import torch

BASE_PATH = Path('/kaggle/input/airbus-ship-detection')
TRAIN_LABEL_PATH = BASE_PATH / 'train_ship_segmentations_v2.csv'
TEST_LABEL_PATH = BASE_PATH / 'sample_submission_v2.csv'
TRAIN_SAMPLE_DIR = BASE_PATH / 'train_v2'
TEST_SAMPLE_DIR = BASE_PATH / 'test_v2'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = 1

COEF_DIF = 14.78 / 100.0

HEIGHT = 256
WIDTH = 256