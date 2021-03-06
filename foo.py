import tensorflow as tf
import numpy as np
import cv2 as cv
from config import *
from model import *


def main():
    temp = input('Enter filename: ')
    f_name = './' + temp + '.png'
    ckpt_state = tf.train.get_checkpoint_state(CHECKPOINT_PATH)
    if not ckpt_state or not ckpt_state.model_checkpoint_path:
        print('No check point files are found!')
        return

    ckpt_files = ckpt_state.all_model_checkpoint_paths
    num_ckpt = len(ckpt_files)
    if num_ckpt < 1:
        print('No check point files are found!')
        return

    low_res_holder = tf.placeholder(tf.float32, shape=[1, INPUT_SIZE, INPUT_SIZE, NUM_CHANNELS])
    inferences = create_model(low_res_holder)

    sess = tf.Session()
    # we still need to initialize all variables even when we use Saver's restore method.
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    cnt = 0
    saver = tf.train.Saver(tf.global_variables())
    for ckpt_file in ckpt_files:
        saver.restore(sess, ckpt_file)  # load the lateast model
        f2x_name = OUTPUT_SAVE_PATH + temp + '_' + str(cnt) + '.png'
        low_res_img = cv.imread(f_name)
        output_size = int(inferences.get_shape()[1])
        input_size = INPUT_SIZE
        available_size = output_size // SCALE_FACTOR
        margin = (input_size - available_size) // 2

        img_rows = low_res_img.shape[0]
        img_cols = low_res_img.shape[1]
        img_chns = low_res_img.shape[2]

        padded_rows = int(img_rows / available_size + 1) * available_size + margin * 2
        padded_cols = int(img_cols / available_size + 1) * available_size + margin * 2
        padded_low_res_img = np.zeros((padded_rows, padded_cols, img_chns), dtype=np.uint8)
        padded_low_res_img[margin: margin + img_rows, margin: margin + img_cols, ...] = low_res_img
        padded_low_res_img = padded_low_res_img.astype(np.float32)
        padded_low_res_img /= 255

        high_res_img = np.zeros((padded_rows * SCALE_FACTOR, padded_cols * SCALE_FACTOR, img_chns), dtype=np.float32)
        low_res_patch = np.zeros((1, input_size, input_size, img_chns), dtype=np.float32)
        for i in range(margin, margin + img_rows, available_size):
            for j in range(margin, margin + img_cols, available_size):
                low_res_patch[0, ...] = padded_low_res_img[i - margin: i - margin + input_size,
                                        j - margin: j - margin + input_size, ...]
                high_res_patch = sess.run(inferences, feed_dict={low_res_holder: low_res_patch})

                out_rows_begin = (i - margin) * SCALE_FACTOR
                out_rows_end = out_rows_begin + output_size
                out_cols_begin = (j - margin) * SCALE_FACTOR
                out_cols_end = out_cols_begin + output_size
                high_res_img[out_rows_begin: out_rows_end, out_cols_begin: out_cols_end, ...] = high_res_patch[0, ...]

        # high_res_img += 0.5
        high_res_img = tf.image.convert_image_dtype(high_res_img, tf.uint8, True)

        high_res_img = high_res_img[:SCALE_FACTOR * img_rows, :SCALE_FACTOR * img_cols, ...]
        cv.imwrite(f2x_name, high_res_img.eval(session=sess))
        print(f2x_name + ' enhance finished.')
        cnt = cnt + 1000

    print('Full image test Finished!')


if __name__ == '__main__':
    main()