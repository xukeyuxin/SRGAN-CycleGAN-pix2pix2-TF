import tensorflow as tf
from utils import reader
from CycleGAN import CycleGAN
from CGAN import CGAN
from SRGAN import SRGAN
from pix2pix import pix2pix
from utils import pix2pix_reader
import numpy as np
import os
import random
import cv2
import sys



sys_input = sys.argv
if(len(sys_input) < 3):
    model = 'CycleGAN'
    type = 'train'
else:
    model = sys_input[1]
    type = sys_input[2]



FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model',model,'model type, default: CycleGAN')

tf.flags.DEFINE_integer('input_size', 64, 'input_size, default: 64')

if(type == 'train'):
    tf.flags.DEFINE_boolean('pretrain', False, 'pretrain, default: True')
else:
    tf.flags.DEFINE_boolean('pretrain', True, 'pretrain, default: True')

tf.flags.DEFINE_string('test_type', 'A', 'test_type, default: A')


tf.flags.DEFINE_float('real_label', 0.9, 'real_label, default: faces.9')
tf.flags.DEFINE_integer('max_to_keep', 5, 'real_label, default: 5')


if(model == 'CycleGAN'):
    tf.flags.DEFINE_float('lr', 1e-4,
                          'initial learning rate for Adam, default: faces.0001')
    tf.flags.DEFINE_integer('batch_size', 5, 'batch size, default: 16')
    tf.flags.DEFINE_integer('epoch', 500, 'test_type, default: 1000')
    tf.flags.DEFINE_string('data_name', 'apple_orange_256', 'test_type, default: apple_orange')
    tf.flags.DEFINE_integer('input_image_weight', 256, 'image size, default: 128')
    tf.flags.DEFINE_integer('input_image_height', 256, 'image size, default: 128')
    tf.flags.DEFINE_integer('output_image_weight', 256, 'image size, default: 256')
    tf.flags.DEFINE_integer('output_image_height', 256, 'image size, default: 256')
    tf.flags.DEFINE_integer('input_image_channels', 3, 'image size, default: 3')

elif(model == 'CGAN'):
    tf.flags.DEFINE_float('lr', 1e-3,
                          'initial learning rate for Adam, default: faces.0001')
    tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 16')
    tf.flags.DEFINE_integer('epoch', 100, 'test_type, default: 1000')
    tf.flags.DEFINE_integer('label_embedding_size', 10, 'label_embedding_size, default: 10')
    tf.flags.DEFINE_integer('input_noise_size', 100, 'input_noise_size, default: 100')
    tf.flags.DEFINE_string('data_name', 'mnist', 'test_type, default: mnist')
    tf.flags.DEFINE_integer('input_image_weight', 28, 'image size, default: 28')
    tf.flags.DEFINE_integer('input_image_height', 28, 'image size, default: 28')
    tf.flags.DEFINE_integer('output_image_weight', 28, 'image size, default: 28')
    tf.flags.DEFINE_integer('output_image_height', 28, 'image size, default: 28')
    tf.flags.DEFINE_integer('input_image_channels', 1, 'image size, default: 3')

elif(model == 'SRGAN'):
    tf.flags.DEFINE_float('lr', 2e-6,
                          'initial learning rate for Adam, default: faces.0001')
    tf.flags.DEFINE_integer('batch_size', 5, 'batch size, default: 16')
    tf.flags.DEFINE_string('data_name', 'MixImage', 'test_type, default: FuzzyImage')
    tf.flags.DEFINE_integer('init_g_epoch', 100, 'test_type, default: 1000')
    tf.flags.DEFINE_integer('epoch', 1000, 'test_type, default: 1000')
    tf.flags.DEFINE_integer('input_image_weight', 96, 'image size, default: 96')
    tf.flags.DEFINE_integer('input_image_height', 96, 'image size, default: 96')
    tf.flags.DEFINE_integer('output_image_weight', 384, 'image size, default: 384')
    tf.flags.DEFINE_integer('output_image_height', 384, 'image size, default: 384')
    tf.flags.DEFINE_integer('input_image_channels', 3, 'image size, default: 3')
elif(model == 'pix2pix'):
    tf.flags.DEFINE_string('data_name', 'maps', 'test_type, default: FuzzyImage')
    tf.flags.DEFINE_float('lr', 1e-4,'learn rate, default: 1e-3')
    tf.flags.DEFINE_integer('batch_size', 16, 'batch size, default: 16')
    tf.flags.DEFINE_integer('epoch', 1000, 'test_type, default: 1000')
    tf.flags.DEFINE_integer('input_image_weight', 256, 'image size, default: 96')
    tf.flags.DEFINE_integer('input_image_height', 256, 'image size, default: 96')
    tf.flags.DEFINE_integer('output_image_weight', 256, 'image size, default: 384')
    tf.flags.DEFINE_integer('output_image_height', 256, 'image size, default: 384')
    tf.flags.DEFINE_integer('input_image_channels', 3, 'image size, default: 3')


tf.flags.DEFINE_integer('lambda1', 10, 'lambda1, default: 10')

tf.flags.DEFINE_float('beta', 1,
                      'momentum term of Adam, default: 1')

model_dict = {'CycleGAN':CycleGAN,
              'CGAN':CGAN,
              'SRGAN':SRGAN,
              'pix2pix':pix2pix}

reader_dict = {'CycleGAN':reader,
              'CGAN':reader,
              'SRGAN':reader,
              'pix2pix':pix2pix_reader}

if __name__=='__main__':
    device_num = "0"
    if(sys.platform == 'darwin'):
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        device = 'gpu'
    with tf.device('%s:%s' % (device,device_num)):
        gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(allow_soft_placement = True, gpu_options = gpu_options)
    with tf.Session( config = config ) as sess:
        Net = model_dict[model](sess, FLAGS, reader_dict[model])
        if(type == 'train'):
            Net.train()
        if(type == 'pretrain'):
            Net.train(pretrain = True)
        elif(type == 'test'):
            Net.test()


