import tensorflow as tf
import numpy as np
import layers as ly
from utils import write_image
from op_base import op_base
import logging
from functools import reduce
from tqdm import tqdm
import random
import os
import cv2
import math


class generator(op_base):
    def __init__(self, name, args):
        op_base.__init__(self, args)
        self.name = name
        self.reuse = False

    def __call__(self, input):
        ### input --> image
        ### input_shape : [ -1, 128,128,3 ]
        input_shape = input.get_shape().as_list()
        input_channel = input_shape[-1]
        with tf.variable_scope(self.name, reuse=self.reuse):
            height = input.get_shape().as_list()[1]

            ### 提取特征 防止图片边缘奇异
            input = tf.pad(input, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
            input = ly.conv2d(input, 64, kernel_size=7, strides=1, name='g_conv2d_0')
            input = ly.bn_layer(input, name='g_bn_0')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 128, kernel_size=3, strides=2, name='g_conv2d_1')
            input = ly.bn_layer(input, name='g_bn_1')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 256, kernel_size=3, strides=2, name='g_conv2d_2')
            input = ly.bn_layer(input, name='g_bn_2')
            input = tf.nn.relu(input)

            ### resnet
            for i in range(8):
                cell = ly.conv2d(input, 256, kernel_size=3, strides=1, name='g_conv2d_res_%s' % i)
                cell = ly.bn_layer(cell, name='g_res_%s' % i)
                cell = tf.nn.relu(cell)
                input = cell

            low_height = math.ceil((height + 6.) / 4)

            input = ly.deconv2d(input, 128, low_height * 2, kernel_size=3, strides=2, name='g_deconv2d_0')
            input = ly.bn_layer(input, name='g_bn_3')
            input = tf.nn.relu(input)

            input = ly.deconv2d(input, 64, low_height * 4  , kernel_size=3, strides=2, name='g_deconv2d_1')
            input = ly.bn_layer(input, name='g_bn_4')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 3, kernel_size=7, strides=1, name='g_conv2d_3')
            input = ly.bn_layer(input, name='g_bn_5')
            input = tf.nn.tanh(input)

            input = tf.image.resize_images(input, [height, height])

        self.reuse = True

        return input

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class discriminator(op_base):
    def __init__(self, name, args):
        op_base.__init__(self, args)
        self.reuse = False
        self.name = name

    def __call__(self, input):
        ### input_shape [-1,128,128,3]
        input_shape = input.get_shape().as_list()

        with tf.variable_scope(self.name, reuse=self.reuse):
            input = ly.conv2d(input, 64, kernel_size=4, strides=2, name='d_conv_0')  # [-1,64,64,64]
            input = ly.bn_layer(input, name='d_bn_0')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 128, kernel_size=4, strides=2, name='d_conv_1')  # [-1,32,32,128]
            input = ly.bn_layer(input, name='d_bn_1')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 256, kernel_size=4, strides=2, name='d_conv_2')  # [-1,16,16,256]
            input = ly.bn_layer(input, name='d_bn_2')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 512, kernel_size=4, strides=2, name='d_conv_3')  # [-1,8,8,512]
            input = ly.bn_layer(input, name='d_bn_3')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 1, kernel_size=4, strides=1, name='d_conv_4', bias=True)
            input = tf.nn.sigmoid(input)

        self.reuse = True
        return input

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class CycleGAN(op_base):
    def __init__(self, sess, args, reader):
        op_base.__init__(self, args)
        self.args = args
        self.sess = sess
        self.Reader = reader(args)

    def cycle_consistency_loss(self):

        G_loss = tf.reduce_mean(tf.abs(self.G(self.fake_x) - self.y))
        F_loss = tf.reduce_mean(tf.abs(self.F(self.fake_y) - self.x))

        return 5 * (G_loss + F_loss)

    def build_model(self):

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_image_weight, self.input_image_height,
                                                   self.input_image_channels])
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_image_weight, self.input_image_height,
                                                   self.input_image_channels])

        self.G = generator('G', self.args)
        self.fake_y = self.G(self.x)

        self.Y_D = discriminator('Y_D', self.args)

        self.F = generator('F', self.args)

        self.fake_x = self.F(self.y)
        self.X_D = discriminator('X_D', self.args)

        cycle_loss = self.cycle_consistency_loss()

        ### g loss
        self.g_loss = tf.reduce_mean(tf.squared_difference(self.real_label, self.Y_D(self.fake_y))) + cycle_loss
        self.f_loss = tf.reduce_mean(tf.squared_difference(self.real_label, self.X_D(self.fake_x))) + cycle_loss

        ### use lsgan d_loss
        self.Y_D_loss = tf.reduce_mean(tf.squared_difference(self.Y_D(self.y), self.real_label)) + tf.reduce_mean(
            tf.square(self.Y_D(self.fake_y)))
        self.X_D_loss = tf.reduce_mean(tf.squared_difference(self.X_D(self.x), self.real_label)) + tf.reduce_mean(
            tf.square(self.X_D(self.fake_x)))

        ### use sigmoid cross entropy
        # self.Y_D_loss = - tf.reduce_mean( tf.log( self.Y_D(self.y ) ) - tf.log(1 - self.Y_D(self.fake_y) ) ) / 2
        # self.X_D_loss = - tf.reduce_mean( tf.log(self.X_D(self.x) ) - tf.log(1 - self.X_D(self.fake_x)) ) / 2

        self.opt_g_Y = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss, var_list=self.G.vars)
        self.opt_f_X = tf.train.AdamOptimizer(self.lr).minimize(self.f_loss, var_list=self.F.vars)

        self.opt_d_Y = tf.train.AdamOptimizer(self.lr).minimize(self.Y_D_loss, var_list=self.Y_D.vars)
        self.opt_d_X = tf.train.AdamOptimizer(self.lr).minimize(self.X_D_loss, var_list=self.X_D.vars)

        with tf.control_dependencies([self.opt_d_Y, self.opt_g_Y, self.opt_d_X, self.opt_f_X]):
            return tf.no_op(name='optimizer')

    def train(self, need_train=True, pretrain=False):

        optimizer = self.build_model()
        saver = tf.train.Saver(max_to_keep = 1)
        self.sess.run(tf.global_variables_initializer())
        if (pretrain):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))
            print('success restore %s' % tf.train.latest_checkpoint(self.model_save_path))
        if (need_train):
            print('start training')
            x_data_path = os.path.join('data', self.model, self.data_name, 'trainA')
            y_data_path = os.path.join('data', self.model, self.data_name, 'trainB')
            x_data_list = os.listdir(x_data_path)
            y_data_list = os.listdir(y_data_path)

            for i in range(self.epoch):
                epoch_size = min(len(x_data_list), len(y_data_list))
                total_g_loss, total_f_loss, total_Y_D_loss, total_X_D_loss, iter = 0., 0., 0., 0., 0
                for batch_time in tqdm(range(epoch_size // self.batch_size)):
                    one_batch_x = self.Reader.build_batch(batch_time, x_data_list, x_data_path)
                    one_batch_y = self.Reader.build_batch(batch_time, y_data_list, y_data_path)
                    _, g_loss, f_loss, Y_D_loss, X_D_loss = self.sess.run(
                        [optimizer, self.g_loss, self.f_loss, self.Y_D_loss, self.X_D_loss],
                        feed_dict={self.x: one_batch_x,
                                   self.y: one_batch_y})
                    iter += 1
                    total_g_loss += g_loss
                    total_f_loss += f_loss
                    total_Y_D_loss += Y_D_loss
                    total_X_D_loss += X_D_loss

                saver.save(self.sess,
                           os.path.join(self.model_save_path, 'checkpoint' + '-' + str(i) + '-' + str(batch_time)))

                print('find g_loss: %s' % (total_g_loss / iter))
                print('find f_loss: %s' % (total_f_loss / iter))
                print('find Y_D_loss: %s' % (total_Y_D_loss / iter))
                print('find X_D_loss: %s' % (total_X_D_loss / iter))

    def test(self):

        test_image_dir = os.path.join('data', self.model, self.data_name, 'test' + self.test_type)
        test_image_list = os.listdir(test_image_dir)
        random.shuffle(test_image_list)
        test_image_content = np.array([])

        for one in test_image_list:
            print(one)
            one = os.path.join(test_image_dir, one)
            if (len(test_image_content) == self.batch_size):
                break
            content = cv2.imread(one)
            if (content.shape[0] != self.input_image_height or content.shape[1] != self.input_image_weight):
                content = cv2.resize(content, (self.input_image_height, self.input_image_weight),
                                     interpolation=cv2.INTER_LINEAR)

            content = content.reshape([1, self.input_image_height, self.input_image_weight, self.input_image_channels])

            if (len(test_image_content) == 0):
                test_image_content = content
            else:
                test_image_content = np.concatenate([test_image_content, content])

        self.train(need_train=False, pretrain=True)

        write_shape = [self.input_image_height, self.input_image_weight, self.input_image_channels]
        if (self.test_type == 'A'):
            generate = self.sess.run(self.fake_y, feed_dict={self.x: test_image_content})
        elif (self.test_type == 'B'):
            generate = self.sess.run(self.fake_x, feed_dict={self.y: test_image_content})
        write_image(generate, self.generate_image_path, write_shape)
