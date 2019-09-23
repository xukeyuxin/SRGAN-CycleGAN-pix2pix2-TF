import tensorflow as tf
import numpy as np
import layers as ly
from utils import write_image
from op_base import op_base
import logging
from functools import reduce
from tqdm import tqdm
import os


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
            input_0 = ly.conv2d(input, 64, kernel_size=4, strides=2, name='g_conv_0')  # [ -1, 128,128,64]
            input_0 = ly.bn_layer(input_0, name='g_bn_0')

            input_1 = ly.conv2d(tf.nn.relu(input_0), 128, kernel_size=4, strides=2, name='g_conv_1')  # [ -1, 64,64,128]
            input_1 = ly.bn_layer(input_1, name='g_bn_1')

            input_2 = ly.conv2d(tf.nn.relu(input_1), 256, kernel_size=4, strides=2, name='g_conv_2')  # [ -1, 32,32,256]
            input_2 = ly.bn_layer(input_2, name='g_bn_2')

            input_3 = ly.conv2d(tf.nn.relu(input_2), 512, kernel_size=4, strides=2, name='g_conv_3')  # [ -1, 16,16,512]
            input_3 = ly.bn_layer(input_3, name='g_bn_3')

            deconv_height = input_3.get_shape().as_list()[1]
            deconv_weight = input_3.get_shape().as_list()[2]

            input_3_de = ly.deconv2d(tf.nn.relu(input_3), 512, deconv_height * 2, kernel_size=4, strides=2,
                                     name='g_deconv_0')  # [-1,32,32,512]
            input_3_de = tf.nn.dropout(input_3_de, 0.2)
            input_3_de = ly.bn_layer(input_3_de, name='g_bn_4')
            input_3_de = tf.concat([input_3_de, input_2], axis=3)

            input_2_de = ly.deconv2d(tf.nn.relu(input_3_de), 256, deconv_height * 4, kernel_size=4, strides=2,
                                     name='g_deconv_1')  # [-1,64,64,256]
            input_2_de = ly.bn_layer(input_2_de, name='g_bn_5')
            input_2_de = tf.concat([input_2_de, input_1], axis=3)

            input_1_de = ly.deconv2d(tf.nn.relu(input_2_de), 128, deconv_height * 8, kernel_size=4, strides=2,
                                     name='g_deconv_2')  # [-1,128,128,128]
            input_1_de = ly.bn_layer(input_1_de, name='g_bn_6')
            input_1_de = tf.concat([input_1_de, input_0], axis=3)


            input_0_de = ly.deconv2d(tf.nn.relu(input_1_de), 3, deconv_height * 16, kernel_size=4, strides=2,
                                     name='g_deconv_3')  # [-1,256,256,3]

            input_0_de = tf.nn.tanh(input_0_de)

        self.reuse = True

        return input_0_de


class discriminator(op_base):
    def __init__(self, name, args):
        op_base.__init__(self, args)
        self.reuse = False
        self.name = name

    def __call__(self, input):
        ### input_shape [-1,128,128,3]
        input_shape = input.get_shape().as_list()

        with tf.variable_scope(self.name, reuse=self.reuse):
            input_0 = ly.conv2d(input, 64, kernel_size=4, strides=2, name='d_conv_0')  # [-1,64,64,64]
            input_0 = tf.nn.relu(ly.bn_layer(input_0, name='d_bn_0'))

            input_1 = ly.conv2d(input_0, 128, kernel_size=4, strides=2, name='d_conv_1')  # [-1,32,32,128]
            input_1 = tf.nn.relu(ly.bn_layer(input_1, name='d_bn_1'))

            input_2 = ly.conv2d(input_1, 256, kernel_size=4, strides=2, name='d_conv_2')  # [-1,16,16,256]
            input_2 = tf.nn.relu(ly.bn_layer(input_2, name='d_bn_2'))

            input_3 = ly.conv2d(input_2, 512, kernel_size=4, strides=1, name='d_conv_3')  # [-1,8,8,512]
            input_3 = tf.nn.relu(ly.bn_layer(input_3, name='d_bn_3'))

            input_3 = ly.conv2d(input_2, 1, kernel_size=4, strides=1, name='d_conv_4',bias = True)

        self.reuse = True
        return input_3


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
        # self.Y_D_loss = - tf.reduce_mean( tf.log( tf.nn.sigmiod(self.Y_D(self.y) ) ) - tf.log(1 - tf.nn.sigmiod(self.Y_D(self.fake_y)) ) ) / 2
        # self.X_D_loss = - tf.reduce_mean( tf.log(self.X_D(self.x) ) - tf.log(1 - self.X_D(self.fake_x)) ) / 2

        self.opt_g_Y = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss)
        self.opt_f_X = tf.train.AdamOptimizer(self.lr).minimize(self.f_loss)

        self.opt_d_Y = tf.train.AdamOptimizer(self.lr).minimize(self.Y_D_loss)
        self.opt_d_X = tf.train.AdamOptimizer(self.lr).minimize(self.X_D_loss)

        with tf.control_dependencies([self.opt_d_Y, self.opt_g_Y, self.opt_d_X, self.opt_f_X]):
            return tf.no_op(name='optimizer')

    def train(self):

        optimizer = self.build_model()
        saver = tf.train.Saver()
        if (self.pretrain):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))
            print('success restore %s' % tf.train.latest_checkpoint(self.model_save_path))
        else:
            print('start training')
            self.sess.run(tf.global_variables_initializer())
            x_data_path = os.path.join('data',self.model, self.data_name, 'trainA')
            y_data_path = os.path.join('data',self.model, self.data_name, 'trainB')
            x_data_list = os.listdir(x_data_path)
            y_data_list = os.listdir(y_data_path)

            for i in range(self.epoch):
                epoch_size = min(len(x_data_list), len(y_data_list))
                for batch_time in tqdm(range(epoch_size // self.batch_size)):
                    one_batch_x = self.Reader.build_batch(batch_time, x_data_list, x_data_path)
                    one_batch_y = self.Reader.build_batch(batch_time, y_data_list, y_data_path)
                    _, g_loss, f_loss, Y_D_loss, X_D_loss = self.sess.run(
                        [optimizer, self.g_loss, self.f_loss, self.Y_D_loss, self.X_D_loss],
                        feed_dict={self.x: one_batch_x,
                                   self.y: one_batch_y})

                saver.save(self.sess,
                           os.path.join(self.model_save_path, 'checkpoint' + '-' + str(i) + '-' + str(batch_time)))
                print(g_loss, f_loss, Y_D_loss, X_D_loss)

    def make_image(self, input_image):
        write_shape = [self.input_image_height, self.input_image_weight, self.input_image_channels]
        if (self.test_type == 'A'):
            generate = self.sess.run(self.fake_y, feed_dict={self.x: input_image})
        elif(self.test_type == 'B'):
            generate = self.sess.run(self.fake_x,feed_dict = {self.y:input_image})
        write_image(generate,self.generate_image_path,write_shape)
