import tensorflow as tf
import numpy as np
import layers as ly
from op_base import op_base
from utils import *
from tqdm import tqdm
import os


class generator(object):
    def __init__(self,name):
        self.name = name
        self.reuse = tf.AUTO_REUSE
    def __call__(self,input):
        with tf.variable_scope(self.name,reuse = self.reuse):
            input = ly.fc(input,7*7*128, name = 'fc_0')
            input = ly.bn_layer(input, name='bn_0')
            input = tf.nn.leaky_relu(input)

            input = tf.reshape(input, (-1, 7, 7, 128))

            input = ly.deconv2d(input,output_channel = 64,output_size = 14,strides = 2,name = 'deconv_0')
            input = ly.bn_layer(input,name = 'bn_1')
            input = tf.nn.leaky_relu(input)

            input = ly.deconv2d(input,output_channel = 1,output_size = 28,strides = 2,name = 'deconv_1')
            input = ly.bn_layer(input,name = 'bn_2')
            input = tf.nn.sigmoid(input) ### 灰度用sigmoid , 彩色用tanh

        return input ## (-1,28,28,1)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)




class discriminator(object):
    def __init__(self,name):
        self.name = name
        self.reuse = tf.AUTO_REUSE
    def __call__(self,input):
        with tf.variable_scope(self.name, reuse = self.reuse):
            input = ly.conv2d(input, 64, strides=2,name = 'conv_0')  ## (-1,14,14,64)
            input = ly.bn_layer(input,name = 'bn_0')
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input, 128, strides=2,name = 'conv_1')  ## (-1,7,7,128)
            input = ly.bn_layer(input, name='bn_1')
            input = tf.nn.leaky_relu(input)

            output_d = ly.fc(input, 1,name = 'fc_0')
            output_d = ly.bn_layer(input, name='bn_2')
            output_d = tf.nn.sigmoid(output_d)

            # output_label = ly.fc(input, 128,name = 'label_fc_0')
            # input = ly.bn_layer(input, name='bn_3')
            # input = tf.nn.leaky_relu(input)
            #
            # output_label = ly.fc(input, 1,name = 'label_fc_1')
            # input = ly.bn_layer(input, name='bn_4')
            # input = tf.nn.sigmoid(input)

        return output_d

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)



class CGAN(op_base):

    def __init__(self,sess,args,reader):
        op_base.__init__(self,args)
        self.sess = sess
        self.Reader = reader(args)

        # data
        self.data = mnist()

    def build_model(self):

        self.x = tf.placeholder(tf.float32,[self.batch_size,self.input_image_weight, self.input_image_height, self.input_image_channels]) ### real image
        self.z = tf.placeholder(tf.float32,[self.batch_size,self.input_noise_size])
        self.y = tf.placeholder(tf.float32,[self.batch_size,self.label_embedding_size])

        # self.g = self.generotor(tf.concat([self.z,self.y],axis = 1))
        self.G = generator('G')
        self.D = discriminator('D')

        self.fake = self.G(tf.concat([self.z,self.y],axis = 1))
        expand_dim_y = tf.reshape(self.y,[self.batch_size,1,1,self.label_embedding_size]) * tf.ones(shape = [self.batch_size,28,28,self.label_embedding_size],dtype = tf.float32 )

        self.d_real = self.D(tf.concat([self.x, expand_dim_y],axis = 3))
        self.d_fake = self.D(tf.concat([self.fake, expand_dim_y],axis = 3))


        safe_log = 1e-12
        ### use cross entropy (safe log)
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake,labels = tf.ones_like(self.d_fake)) )
        # self.g_loss = - tf.reduce_mean( tf.log( self.d_fake + safe_log) )

        # ### use lsgan
        # real_weight = 0.9
        # self.g_loss = tf.reduce_mean( tf.squared_difference(real_weight,d_fake) )
        # # self.g_loss = tf.reduce_mean( tf.squared_difference(real_weight,d_fake) ) + 5 * tf.squared_difference(self.fake,self.x)

        ### use cross entropy (safe log)
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake,labels = tf.zeros_like(self.d_fake)) ) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_real,labels = tf.ones_like(self.d_real)) )
        # self.d_loss = -tf.reduce_mean(tf.log( 1 - self.d_fake + safe_log )) - tf.reduce_mean( tf.log(self.d_real + safe_log ))

        # ### use lsgan
        # self.d_loss = tf.reduce_mean(tf.square(self.d_fake)) + tf.reduce_mean(tf.suqared_diffrence(self.d_real - real_weight))

        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss)
        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss)


        ### check loss
        self.check_d_real = tf.reduce_mean(self.d_real)
        self.check_d_fake = tf.reduce_mean(self.d_fake)

        return self.opt_g, self.opt_d

        # with tf.control_dependencies([self.opt_g, self.opt_d]):
        #     return tf.no_op(name='optimizer')

    def z_sample(self):
        return np.random.uniform(-1.,1.,size = [self.batch_size,self.input_noise_size])

    def train(self):
        opt_g, opt_d = self.build_model()
        # optimizer = self.build_model()
        saver = tf.train.Saver()
        if(self.pretrain):
            saver.restore(self.sess,tf.train.latest_checkpoint(self.model_save_path))
        else:
            self.sess.run(tf.global_variables_initializer())
            print('start train')
            for num in range(1000000):
                X_b, y_b = self.data(self.batch_size)

                self.sess.run(
                    opt_d,
                    feed_dict={self.x:X_b,self.y:y_b,self.z:self.z_sample()} )
                self.sess.run(
                    opt_g,
                    feed_dict={self.y: y_b, self.z: self.z_sample()}
                )
                # for i in range(1):
                #     _,d_loss,g_loss,fake = self.sess.run([optimizer,self.d_loss,self.g_loss,self.fake],feed_dict = {self.x:X_b,self.y:y_b,self.z:self.z_sample()})

                if(num % 1000 == 0):

                    g_loss,d_loss,fake = self.sess.run(
                        [self.g_loss,self.d_loss,self.fake],
                        feed_dict={self.x: X_b, self.y: y_b, self.z: self.z_sample()})

                    write_shape = [self.input_image_height, self.input_image_weight, self.input_image_channels]
                    write_image_gray(fake, self.generate_image_path, write_shape)
                    saver.save(self.sess,os.path.join(self.model_save_path, 'checkpoint' + '-' + str(num)))
                    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(num, d_loss, g_loss))


    def test(self):
        self.train()
        X_b, y_b = self.data(self.batch_size)
        fake = self.sess.run(
                self.fake,
                feed_dict={self.x: X_b, self.y: y_b, self.z: self.z_sample()})
        write_shape = [self.input_image_height, self.input_image_weight, self.input_image_channels]
        write_image_gray(fake,self.generate_image_path,write_shape)









