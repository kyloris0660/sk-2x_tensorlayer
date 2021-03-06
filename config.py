# parameters for model and training
INPUT_SIZE = 36
NUM_CHANNELS = 3
PATCH_SIZE = 80
SCALE_FACTOR = 2
LABEL_SIZE = SCALE_FACTOR * INPUT_SIZE
JPEG_NOISE_LEVEL = 1
GAUSSIAN_NOISE_STD = 0.01
BATCH_SIZE = 16
NUM_EPOCHS = 400000
MODEL_NAME = 'vgg12'
# train dataset path and etc
TRAIN_PATH = 'E:/image_data/data_train/'
TEST_PATH = 'E:/image_data/data_test/'
TRAINING_SUMMARY_PATH = 'E:/image_data/Training_summary/'
CHECKPOINT_PATH = 'E:/image_data/checkpoint/'
INFERENCE_SAVE_PATH = 'E:/image_data/inference/'
OUTPUT_SAVE_PATH = './save/'
