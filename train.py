import tensorflow as tf
import tensorlayer as tl
import time

from input_data import *
from config import *
from model import *


def main():
    low_res_holder = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, NUM_CHANNELS])
    high_res_holder = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, LABEL_SIZE, LABEL_SIZE, NUM_CHANNELS])
    inference = create_model(low_res_holder)
    training_loss = s_mse_loss(inference, high_res_holder)
    validation_loss = s_mse_loss(inference, high_res_holder, name='validation_loss')
    tf.summary.scalar('training_loss', training_loss)
    tf.summary.scalar('validation_loss', validation_loss)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.inverse_time_decay(0.001, global_step, 10000, 2)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(training_loss, global_step=global_step)
    low_res_batch, high_res_batch = generate_train_queue(TRAIN_PATH)
    low_res_eval, high_res_eval = generate_train_queue(TEST_PATH)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)

    # Restore the saver
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TRAINING_SUMMARY_PATH, sess.graph)

    for step in range(0, NUM_EPOCHS):
        start_time = time.time()
        low_res_images, high_res_images = sess.run([low_res_batch, high_res_batch])
        feed_dict = {low_res_holder: low_res_images, high_res_holder: high_res_images}
        _, batch_loss = sess.run([train_step, training_loss], feed_dict=feed_dict)
        duration = time.time() - start_time

        if step % 100 == 0:
            num_examples_per_step = BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = 'step %d, batch_loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (step, batch_loss, examples_per_sec, sec_per_batch))

        if step % 1000 == 0:
            low_res_images, high_res_images = sess.run([low_res_eval, high_res_eval])
            feed_dict = {low_res_holder: low_res_images, high_res_holder: high_res_images}
            batch_loss = sess.run(validation_loss, feed_dict=feed_dict)
            print('step %d, validation loss = %.3f' % (step, batch_loss))

            summary = sess.run(merged_summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary)

        if step % 10000 == 0 or step == NUM_EPOCHS:
            saver.save(sess, join(CHECKPOINT_PATH, 'model.ckpt'), global_step=step)

        print('=========training finished=========')


if __name__ == '__main__':
    main()
