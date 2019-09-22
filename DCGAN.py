import tensorflow as tf
import numpy as np
import layers as ly
from op_base import op_base


class GAN(op_base):
    def __init__(self, sess, args):
        op_base.__init__(self, sess, args)

    def conv_layer(input, output_size, name, kernel_size=3, strides=1, need_resize=0, need_bn=True, need_dropout=0,
                   activate='relu'):

        if (need_resize):
            input = ly.resize_nn(input, need_resize)

        input = ly.conv2d(input, output_size, kernel_size, strides, name=name)

        if (need_bn):
            input = tf.layers.batch_normalization(input, training=True)

        if (activate == 'relu'):
            input = tf.nn.relu(input)
        elif(activate == 'leaky_relu')
            input = tf.nn.leaky_relu(input)
        elif (activate == 'tanh'):
            input = tf.nn.tanh(input)

        if (need_dropout):
            input = tf.nn.dropout(need_dropout)

        return input

    def generator(self, input, name=None, need_bn=True, activate='relu'):

        # self.input_size = 128
        # image_size = 128 * 128
        base_input_size = 8 * 8 * self.input_size
        input = ly.fc(input, base_input_size, name='fc_g_0')
        input = tf.rehshape(input, [self.batch_size, 8, 8, self.input_size])

        input = self.conv_layer(input, 128, 'cond2d_g_0', need_resize=16)
        input = self.conv_layer(input, 64, 'cond2d_g_1', need_resize=32)
        input = self.conv_layer(input, 32, 'cond2d_g_2', need_resize=64)
        input = self.conv_layer(input, 3, 'cond2d_g_3', need_resize=128, need_bn=False, activate='tanh')
        input = tf.reshape(input, [self.batch_size, 128, 128, 3])

        return input

    def discriminator(self, input):

        input = self.conv_layer(input, 64, 'cond2d_d_0', strides=2, need_dropout=0.2,activate = 'leaky_relu')
        input = self.conv_layer(input, 128, 'cond2d_d_1', strides=2, need_dropout=0.2,activate = 'leaky_relu')
        input = self.conv_layer(input, 256, 'cond2d_d_2', strides=2, need_dropout=0.2,activate = 'leaky_relu')

        flatten_shape = sum(input.get_shape().as_list()[1:])
        input = tf.reshape(input, [-1,flatten_shape])
        input = ly.fc(input,1)

        return input

    def model(self):
        '''
        ### g_loss
        g = generator()
        dis_fake = discriminator(g)
        g_loss = l1_loss( dis_fake , real_image)

        ### d_loss
        dis_real = img
        real_d_loss = sigmoid( dis_real, target = tf.ones )
        fake_d_loss = sigmoid( dis_fake, target = tf.zeros )

        :return: alpha（ g_loss ） +  beta (real_d_loss + fake_d_loss)
        '''

        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_noise_size])
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_image_weight, self.input_image_height,
                                                   self.input_image_channels])

        gen_x = self.generator(self.x)

        dis_x = self.discriminator(gen_x)
        dis_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_x, labels=tf.zeros(shape=[self.batch_size, 1]))

        dis_y = self.discriminator(self.y)
        dis_y_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_y, labels=tf.ones(shape=[self.batch_size, 1]))

        d_loss = self.alpha * (dis_x_loss + dis_y_loss)
        g_loss = self.beta * ly.l1_loss(gen_x - tf.ones(shape=[self.batch_size, 1]))

        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(d_loss)
        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(g_loss)

        self.sess.run(tf.global_variables_initializer())

    def train(self):
        pass
