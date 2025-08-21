from pathlib import Path

BASE_PATH = Path('/kaggle/input/airbus-ship-detection')
TRAIN_LABEL_PATH = BASE_PATH / 'train_ship_segmentations_v2.csv'
TEST_LABEL_PATH = BASE_PATH / 'sample_submission_v2.csv'
TRAIN_SAMPLE_DIR = BASE_PATH / 'train_v2'
TEST_SAMPLE_DIR = BASE_PATH / 'test_v2'