# parameters for model and training
INPUT_SIZE = 28
NUM_CHANNELS = 3
PATCH_SIZE = 80
SCALE_FACTOR = 2
LABEL_SIZE = SCALE_FACTOR * INPUT_SIZE
JPEG_NOICE_LEVEL = 1
GAUSSIAN_NOISE_STD = 0.01
BATCH_SIZE = 16
NUM_STEPS = 1000000
# train dataset path and etc
TRAIN_PATH = '/Users/kyloris/Projects/image_data/data_train'
TEST_PATH = '/Users/kyloris/Projects/image_data/data_test'
TRAINING_SUMMARY_PATH = '/Users/kyloris/Projects/image_data/Training_summary'
CHECKPOINT_PATH = '/Users/kyloris/Projects/image_data/checkpoint'
