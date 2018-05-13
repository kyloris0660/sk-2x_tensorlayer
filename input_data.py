import tensorflow as tf
from config import *
from os.path import join


def generate_train_queue(data_path):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(join(data_path, '*.png')))
