import tensorflow as tf
from utils import reader
from CycleGAN import CycleGAN
import numpy as np
import os
import random
import cv2
import sys


type = sys.argv[1]

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 16')
tf.flags.DEFINE_integer('input_size', 64, 'input_size, default: 64')


if(type == 'train'):
    tf.flags.DEFINE_boolean('pretrain', False, 'pretrain, default: True')
else:
    tf.flags.DEFINE_boolean('pretrain', True, 'pretrain, default: True')

tf.flags.DEFINE_string('test_type', 'B', 'test_type, default: A')

tf.flags.DEFINE_integer('epoch', 1, 'test_type, default: 1000')
tf.flags.DEFINE_float('real_label', 0.9, 'real_label, default: 0.9')

tf.flags.DEFINE_string('data_name', 'apple_orange_256', 'test_type, default: apple_orange')


tf.flags.DEFINE_integer('embedding_size', 10, 'embedding_size, default: 10')
tf.flags.DEFINE_integer('input_noise_size', 100, 'input_noise_size, default: 100')

tf.flags.DEFINE_integer('input_image_weight', 256, 'image size, default: 128')
tf.flags.DEFINE_integer('input_image_height', 256, 'image size, default: 128')
tf.flags.DEFINE_integer('input_image_channels', 3, 'image size, default: 3')

tf.flags.DEFINE_integer('lambda1', 10, 'lambda1, default: 10')

tf.flags.DEFINE_float('lr', 1e-4,
                      'initial learning rate for Adam, default: 0.0001')
tf.flags.DEFINE_float('beta', 1,
                      'momentum term of Adam, default: 1')




def trainCycleGAN():
    with tf.Session() as sess:
        Net = CycleGAN(sess,FLAGS,reader)
        Net.train()

def testCycleGAN():

    with tf.Session() as sess:
        Net = CycleGAN(sess,FLAGS,reader)
        test_image_dir = os.path.join('data',FLAGS.data_name,'test' + FLAGS.test_type)
        test_image_list =  os.listdir(test_image_dir)
        random.shuffle(test_image_list)
        test_image_content = np.array([])

        for one in test_image_list:
            print(one)
            one = os.path.join(test_image_dir, one)
            if(len(test_image_content) == FLAGS.batch_size):
                break
            content = cv2.imread(one)
            if(content.shape[0] != FLAGS.input_image_height or content.shape[1] != FLAGS.input_image_weight):
                content = cv2.resize(content,(FLAGS.input_image_height,FLAGS.input_image_weight),interpolation = cv2.INTER_LINEAR)

            content = content.reshape([1,FLAGS.input_image_height,FLAGS.input_image_weight,FLAGS.input_image_channels])

            if(len(test_image_content) == 0 ):
                test_image_content = content
            else:
                test_image_content = np.concatenate([test_image_content,content])

        Net.train()
        Net.make_image(test_image_content)

if __name__ == '__main__':
    if(type == 'train'):
        trainCycleGAN()
    else:
        testCycleGAN()

