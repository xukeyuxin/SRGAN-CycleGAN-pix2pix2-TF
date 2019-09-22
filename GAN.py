import tensorflow as tf
import numpy as np
import layers as ly
from op_base import op_base

class GAN(op_base):
    def __init__(self,sess,args):
        op_base.__init__(self,sess,args)


    def generator(self,input):
        # self.input_size = 64
        base_input_size = 8 * 8 * self.input_size
        input = tf.nn.leaky_relu(ly.fc(input,base_input_size,name = 'fc_g_0'))
        input = tf.layers.batch_normalization(input, training= True)


        base_input_size = 32 * 32 * self.input_size
        input = tf.nn.leaky_relu(ly.fc(input,base_input_size,name = 'fc_g_1'))
        input = tf.layers.batch_normalization(input, training= True)


        base_input_size =  128 * 128 * self.input_size
        input = tf.nn.leaky_relu(ly.fc(input,base_input_size,name = 'fc_g_2'))
        input = tf.layers.batch_normalization(input, training= True)

        base_input_size = 128 * 128 * 3
        input = tf.nn.tanh(ly.fc(input,base_input_size,name = 'fc_g_3'))
        input = tf.reshape(input,[self.batch_size,128,128,3])

        return input


    def discriminator(self,input):
        base_input_size = sum(input.get_shape().as_list()[1:])
        input = tf.reshape(input,[-1,base_input_size])

        base_input_size = 512
        input = tf.nn.leaky_relu(ly.fc(input,base_input_size))

        base_input_size = 256
        input = tf.nn.leaky_relu(ly.fc(input,base_input_size))

        base_input_size = 1
        input = ly.fc(input,base_input_size)

        return input

    def model(self):
        '''
        ### g_loss
        g = generator()
        dis_fake = discriminator(g)
       `g_loss = l1_loss( dis_fake , real_image)

        ### d_loss
        dis_real = img
        real_d_loss = sigmoid( dis_real, target = tf.ones )
        fake_d_loss = sigmoid( dis_fake, target = tf.zeros )

        :return: alpha（ g_loss ） +  beta (real_d_loss + fake_d_loss)
        '''

        self.x = tf.placeholder(tf.float32,shape = [self.batch_size, self.input_noise_size])
        self.y = tf.placeholder(tf.float32,shape = [self.batch_size, self.input_image_weight, self.input_image_height, self.input_image_channels])

        gen_x = self.generator(self.x)

        dis_x = self.discriminator(gen_x)
        dis_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = dis_x,labels = tf.zeros( shape = [self.batch_size,1] ) )

        dis_y = self.discriminator(self.y)
        dis_y_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits= dis_y, labels=tf.ones(shape=[self.batch_size, 1]))

        d_loss = self.alpha * ( dis_x_loss + dis_y_loss )
        g_loss = self.beta * ly.l1_loss( gen_x - tf.ones( shape = [self.batch_size,1] ) )

        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(d_loss)
        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(g_loss)

        self.sess.run(tf.global_variables_initializer())

    def train(self):
        pass










