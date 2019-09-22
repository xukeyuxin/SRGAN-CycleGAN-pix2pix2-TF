import tensorflow as tf
import numpy as np
import layers as ly
from op_base import op_base


class CGAN(op_base):

    def __init__(self,sess,args):
        op_base.__init__(self,sess,args)

    def generotor(self,input):

        ### input --> noise
        input_size = sum(input.get_shape().as_list()[1:])

        #### add labels
        embedding_size = 10
        input_noise_label_weight = ly.xavier_init([input_size,embedding_size] , init_type = 'normal',trainable = True)
        input_noise_label_bias = ly.xavier_init(embedding_size,init_type = 'zeros',trainable = True)
        input = tf.add_bias( tf.matmul(input,input_noise_label_weight) , input_noise_label_bias )

        input = ly.fc(input,128 * 128 * 3)
        input = tf.reshape(input,[-1,128,128,3])

        return input

    def discriminator(self,input):
        embedding_size = self.embedding_size
        input_size = sum(input.get_shape().as_list()[1:])
        input_image_label_weight = ly.xavier_init([input_size,embedding_size], init_type = 'normal',trainable = True)
        input_image_label_bias = ly.xavier_init(embedding_size, init_type = 'zeros', trainable = True)
        input = tf.add_bias( tf.matmul(input, input_image_label_weight), input_image_label_bias)

        input = tf.reshape(input,[-1,input_size])
        input = ly.fc(input,512)
        input = ly.fc(input, 256)
        input = ly.fc(input, 1)

        return input

    def model(self):

        self.x = tf.placeholder(tf.float32,[self.batch_size,self.input_noise_size])
        self.y = tf.placeholder(tf.float32,[self.batch_size,self.input_image_weight, self.input_image_height, self.input_image_channels])

        self.g = self.generotor(self.x)
        self.d_x = self.discriminator(self.x)
        self.d_y = self.discriminator(self.y)

        self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros([self.batch_size,1], logits = self.g ))

        self.d_x_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros([self.batch_size, 1], logits=self.d_x))
        self.d_y_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones([self.batch_size, 1], logits=self.d_y))

        self.d_loss = self.d_x_loss + self.d_y_loss

        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss)
        self.opt_l = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss)








