import tensorflow as tf
import numpy as np
import layers as ly
from op_base import op_base
import os
from functools import reduce
from tqdm import tqdm
import random
from utils import *

class generator(object):
    def __init__(self, name):
        self.name = name
        self.reuse = tf.AUTO_REUSE

    def __call__(self, input):
        height = input.get_shape().as_list()[1]

        with tf.variable_scope(self.name, reuse=self.reuse):
            input = ly.conv2d(input, 64, name='g_conv2d_0')
            input = ly.bn_layer(input, name='g_bn_0')
            input = tf.nn.relu(input)

            ### resnet
            for i in range(16):
                cell = ly.conv2d(input, 64, name='g_res_conv2d_%s' % i)
                cell = ly.bn_layer(cell, name='g_res_bn_%s' % i)
                cell = input + cell
                input = cell

            ### deconv2d
            input = ly.conv2d(input, 256, name='g_conv2d_1')
            input = ly.deconv2d(input, 128, height * 2, strides=2, name='g_deconv2d_0')
            input = tf.nn.relu(input)

            input = ly.conv2d(input, 128, name='g_conv2d_2')
            input = ly.deconv2d(input, 64, height * 4, strides=2, name='g_deconv2d_1')
            input = tf.nn.relu(input)

            # ### Upsampling
            # input = ly.conv2d(input,256,name = 'g_conv2d_1')
            #
            # input = ly.Upsampling(input,height * 2)
            # input = ly.conv2d(input, 128,name = 'g_conv2d_2')
            # input = ly.bn_layer(input,name = 'g_bn_2')
            #
            # input = ly.Upsampling(input,height * 4)
            # input = ly.conv2d(input, 64,name = 'g_conv2d_3')
            # input = ly.bn_layer(input,name = 'g_bn_3')

            input = ly.conv2d(input, 3, name='g_conv2d_last')
            input = tf.nn.tanh(input)

            return input

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class discriminator(object):
    def __init__(self, name):
        self.name = name
        self.reuse = tf.AUTO_REUSE

    def __call__(self, input):
        ### simple
        with tf.variable_scope(self.name, reuse=self.reuse):
            input = ly.conv2d(input, 64, name='d_conv2d_0')

            input = ly.conv2d(input, 64, strides=2, name='d_conv2d_1')
            input = ly.bn_layer(input, name='d_bn_1', gamma_initializer=tf.random_normal_initializer(1., 0.02))
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input, 128, strides=1, name='d_conv2d_2')
            input = ly.bn_layer(input, name='d_bn_2', gamma_initializer=tf.random_normal_initializer(1., 0.02))
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input, 128, strides=2, name='d_conv2d_3')
            input = ly.bn_layer(input, name='d_bn_3', gamma_initializer=tf.random_normal_initializer(1., 0.02))
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input, 256, strides=1, name='d_conv2d_4')
            input = ly.bn_layer(input, name='d_bn_4', gamma_initializer=tf.random_normal_initializer(1., 0.02))
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input, 256, strides=2, name='d_conv2d_5')
            input = ly.bn_layer(input, name='d_bn_5', gamma_initializer=tf.random_normal_initializer(1., 0.02))
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input, 512, strides=1, name='d_conv2d_6')
            input = ly.bn_layer(input, name='d_bn_6', gamma_initializer=tf.random_normal_initializer(1., 0.02))
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input, 512, strides=2, name='d_conv2d_7')
            input = ly.bn_layer(input, name='d_bn_7', gamma_initializer=tf.random_normal_initializer(1., 0.02))
            input = tf.nn.leaky_relu(input)

            print(input.shape)
            input = ly.fc(input, 1024, name='d_fc_0')
            input = tf.nn.leaky_relu(input)

            input = ly.fc(input, 1, name='d_fc_1')
            input = tf.nn.sigmoid(input)

            return input
        ### complex

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class SRGAN(op_base):
    def __init__(self, sess, args,reader):
        op_base.__init__(self, args)
        self.sess = sess
        self.Reader = reader

    def build_model(self):

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_image_weight, self.input_image_height,
                                                   self.input_image_channels])
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_image_weight, self.output_image_height,
                                                   self.input_image_channels])

        self.input_image = tf.placeholder(tf.float32,
                                    shape=[1, self.input_image_weight, self.input_image_height,
                                           self.input_image_channels])

        self.G = generator('G')
        self.D = discriminator('D')
        self.VGG19 = ly.vgg19('VGG19')

        self.fake = self.G(self.x)
        fake_shape = self.fake.get_shape().as_list()

        self.d_fake = self.D(self.fake)
        self.d_real = self.D(self.y)



        safe_log = 1e-12
        self.d_loss = tf.reduce_mean(- tf.log(self.d_real + safe_log) - tf.log(1 - self.d_fake + safe_log))

        self.g_d_loss = 1e-3 * tf.reduce_mean(- tf.log(self.d_fake + safe_log))

        mean_square_content = tf.square(self.fake - self.y)
        mean_square_content = tf.reshape(mean_square_content,
                                         [self.batch_size, reduce(lambda x,y: x * y, fake_shape[1:])])
        self.g_mean_square = 1 * tf.reduce_mean(tf.reduce_sum(mean_square_content, axis=1))

        vgg_loss = self.VGG19(self.fake) - self.VGG19(self.y)
        vgg_loss_shape = vgg_loss.get_shape().as_list()
        mean_square_feature = tf.square(vgg_loss)
        mean_square_feature = tf.reshape(mean_square_feature,
                                         [self.batch_size, reduce(lambda x,y: x * y, vgg_loss_shape[1:])])
        self.vgg_mean_square = 2e-6 * tf.reduce_mean(tf.reduce_sum(mean_square_feature, axis=1))

        self.g_loss = self.g_d_loss + self.g_mean_square + self.vgg_mean_square


        self.opt_mse = tf.train.AdamOptimizer(self.lr).minimize(self.g_mean_square,var_list = self.G.vars)
        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss,var_list = self.D.vars)
        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss,var_list = self.G.vars)

        # with tf.control_dependencies([self.opt_d,self.opt_g]):
        #     return tf.no_op(name = 'optimizer')



    def save_params(self,values):
        print('start find')
        for one in values:
            print(one.name)
        return

    def train(self,need_train = True,pretrain = False):

        self.build_model()
        saver = tf.train.Saver(max_to_keep = 1)
        self.sess.run(tf.global_variables_initializer())
        if (pretrain):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.model_init_g_save_path))
            print('success restore %s' % tf.train.latest_checkpoint(self.model_init_g_save_path))
        if(need_train):
            print('start train')
            x_data_path = os.path.join('data',self.model, self.data_name, 'FuzzyImage')
            y_data_path = os.path.join('data',self.model, self.data_name, 'ClearImage')
            x_data_list = os.listdir(x_data_path)
            y_data_list = os.listdir(y_data_path)

            #### train g_init make mse_loss smaller
            # for i in range(self.init_g_epoch):
            #     mse_total,iter = 0, 0
            #     epoch_size = min(len(x_data_list), len(y_data_list))
            #     for batch_time in tqdm(range(epoch_size // self.batch_size)):
            #         one_batch_x = self.Reader.build_batch(self,batch_time, x_data_list, x_data_path)
            #         one_batch_y = self.Reader.build_batch(self,batch_time, y_data_list, y_data_path)
            #
            #         g_mean_square,fake,_ = self.sess.run([self.g_mean_square,self.fake,self.opt_g],
            #                             feed_dict={self.x: one_batch_x,self.y: one_batch_y})
            #
            #         mse_total += g_mean_square
            #         iter += 1
            #         print(g_mean_square)
            # print('mean : %s' % (mse_total // iter))
            # saver.save(self.sess,
            #            os.path.join(self.model_init_g_save_path,
            #                         'checkpoint_init_g' + '-' + str(i) + '-' + str(batch_time)))
            ##### mse loss 3262


            #### train g,d with SRGAN make g_loss smaller
            for i in range(self.epoch):
                epoch_size = min(len(x_data_list), len(y_data_list))
                total_d_loss,total_g_d_loss,total_mse_loss,total_vgg_loss,iter = 0.,0.,0.,0.,0
                for batch_time in tqdm(range(epoch_size // self.batch_size)):
                    one_batch_x = self.Reader.build_batch(self,batch_time, x_data_list, x_data_path)
                    one_batch_y = self.Reader.build_batch(self,batch_time, y_data_list, y_data_path)

                    _, d_loss, d_fake, d_real = self.sess.run([self.opt_d, self.d_loss,self.d_fake,self.d_real], feed_dict={self.x: one_batch_x,self.y: one_batch_y})

                    _, g_loss,g_d_loss,g_mse,g_vgg = self.sess.run(
                        [self.opt_g, self.g_loss,self.g_d_loss,self.g_mean_square,self.vgg_mean_square],
                        feed_dict={self.x: one_batch_x,
                                   self.y: one_batch_y})

                    total_d_loss += d_loss
                    total_g_d_loss += g_d_loss
                    total_mse_loss += g_mse
                    total_vgg_loss += g_vgg
                    iter += 1

                    print('find d-loss: %s' % (total_d_loss / iter))
                    print('find g_d-loss: %s' % (total_g_d_loss / iter))
                    print('find g_mse-loss: %s' % (total_mse_loss / iter))
                    print('find g_vgg-loss: %s' % (total_vgg_loss / iter))

                saver.save(self.sess,
                           os.path.join(self.model_save_path, 'checkpoint' + '-' + str(i) + '-' + str(batch_time)))

    def test(self):

        # x_data_path = os.path.join('data', self.model, self.data_name, 'FuzzyImage')
        # x_data_list = os.listdir(x_data_path)
        # choose_batch_iter = random.choice( range(len(x_data_list) // self.batch_size ) )
        # one_batch_x = self.Reader.build_batch(self,choose_batch_iter, x_data_list, x_data_path)

        test_image = test_one_image('fuzzy.png')
        self.train(pretrain=True, need_train=False)
        # self.batch_size = len(test_image)
        # fake = self.sess.run(self.fake,feed_dict = {self.x:test_image})
        #
        # write_shape = [self.output_image_height,self.output_image_weight,self.input_image_channels]
        # write_image(fake,self.generate_image_path,write_shape)


