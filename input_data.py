import tensorflow as tf
from config import *
from os.path import join


def generate_train_queue(data_path):
    # Add image file to queue
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(join(data_path, '*.png')))
    file_reader = tf.WholeFileReader()
    _, image_file = file_reader.read(filename_queue)
    patch = tf.image.decode_png(image_file, NUM_CHANNELS)
    # Set the static shape of the patch node
    patch.set_shape([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS])
    patch = tf.image.convert_image_dtype(patch, dtype=tf.float32)
    patch = tf.image.random_flip_left_right(patch)
    crop_margin = PATCH_SIZE - LABEL_SIZE
    # Crop image from patch
    if crop_margin >= 1:
        high_res_patch = tf.random_crop(patch, (LABEL_SIZE, LABEL_SIZE, NUM_CHANNELS))
    # methods of cropping
    downscale_size = [INPUT_SIZE, INPUT_SIZE]
    resize_nn = lambda: tf.image.resize_nearest_neighbor([high_res_patch], downscale_size, True)
    resize_area = lambda: tf.image.resize_area([high_res_patch], downscale_size, True)
    resize_cubic = lambda: tf.image.resize_bicubic([high_res_patch], downscale_size, True)
    r = tf.random_uniform([], 0, 3, dtype=tf.int32)
    low_res_patch = tf.case({tf.equal(r, 0): resize_nn, tf.equal(r, 1): resize_area}, default=resize_cubic)[0]
    # Add JPEG noice
    if JPEG_NOICE_LEVEL > 0:
        low_res_patch = tf.image.convert_image_dtype(low_res_patch, dtype=tf.uint8, saturate=True)
        jpeg_quality = 100 - 5 * JPEG_NOICE_LEVEL
        jpeg_code = tf.image.encode_jpeg(low_res_patch, quality=jpeg_quality)
        low_res_patch = tf.image.decode_jpeg(jpeg_code)
        low_res_patch = tf.image.convert_image_dtype(low_res_patch, dtype=tf.float32)




