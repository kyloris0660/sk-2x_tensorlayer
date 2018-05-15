from input_data import *
from model import *

import cv2 as cv
import tensorflow as tf
import os
import numpy as np


def main():
    ckpt_state = tf.train.get_checkpoint_state(CHECKPOINT_PATH)
    if not ckpt_state or not ckpt_state.model_checkpoint_path:
        print('No check point files are found!')
        return

    ckpt_files = ckpt_state.all_model_checkpoint_paths
    num_ckpt = len(ckpt_files)
    if num_ckpt < 1:
        print('No check point files are found!')
        return

    low_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHANNELS])
    high_res_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, LABEL_SIZE, LABEL_SIZE, NUM_CHANNELS])

    inferences = create_model(low_res_holder)
    low_res_batch, high_res_batch = generate_test_queue(TEST_PATH)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())
    tf.train.start_queue_runners(sess=sess)
    cnt = 0
    for ckpt_file in ckpt_files:
        print('=========================================================')
        print('=========================================================')
        print('Using models of ' + ckpt_file + ' to generate some patches.')
        saver.restore(sess, ckpt_file)

        for k in range(4):
            low_res_images, high_res_images = sess.run([low_res_batch, high_res_batch])
            feed_dict = {low_res_holder: low_res_images, high_res_holder: high_res_images}
            inference_patches = sess.run(inferences, feed_dict=feed_dict)

            if not os.path.exists(INFERENCE_SAVE_PATH):
                os.mkdir(INFERENCE_SAVE_PATH)

            for i in range(BATCH_SIZE):
                low_res_input = low_res_images[i, ...]  # INPUT_SIZE x INPUT_SIZE
                ground_truth = high_res_images[i, ...]  # LABEL_SIZE x LABEL_SIZE
                inference = inference_patches[i, ...]

                crop_begin = (ground_truth.shape[0] - inference.shape[0]) // 2
                crop_end = crop_begin + inference.shape[0]
                ground_truth = ground_truth[crop_begin: crop_end, crop_begin: crop_end, ...]
                low_res_input = cv.resize(low_res_input, (LABEL_SIZE, LABEL_SIZE), interpolation=cv.INTER_CUBIC)
                low_res_input = low_res_input[crop_begin: crop_end, crop_begin: crop_end, ...]
                patch_pair = np.hstack((low_res_input, inference, ground_truth))

                # patch_pair += 0.5
                patch_pair = tf.image.convert_image_dtype(patch_pair, tf.uint8, True)

                save_name = 'inference_%d_%d_%d.png' % (k, i, cnt)
                cv.imwrite(join(INFERENCE_SAVE_PATH, save_name), patch_pair.eval(session=sess))
        cnt = cnt + 1000
    print('Test Finished!')


if __name__ == '__main__':
    main()
