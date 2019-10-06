import os
import tensorflow as tf
import numpy as np
from op_base import op_base
import layers as ly
from utils import *
from tqdm import tqdm


class generator(object):
    def __init__(self,name):
        self.name = name
        self.reuse = tf.AUTO_REUSE

    def __call__(self,input):# 256 * 256 * 3
        with tf.variable_scope(self.name, reuse = self.reuse):

            e0 = ly.conv2d(input,64,strides = 2,name = 'g_conv2d_0') # 128 * 128 * 64
            e0 = ly.bn_layer(e0,name = 'g_bn_0')
            e0 = tf.nn.leaky_relu(e0)

            e1 = ly.conv2d(e0,128,strides = 2,name = 'g_conv2d_1')  # 64 * 64 * 128
            e1 = ly.bn_layer(e1,name = 'g_bn_1')
            e1 = tf.nn.leaky_relu(e1)

            e2 = ly.conv2d(e1,256,strides = 2,name = 'g_conv2d_2')# 32 * 32 * 256
            e2 = ly.bn_layer(e2,name = 'g_bn_2')
            e2 = tf.nn.leaky_relu(e2)

            e3 = ly.conv2d(e2,512,strides = 2,name = 'g_conv2d_3')# 16 * 16 * 512
            e3 = ly.bn_layer(e3,name = 'g_bn_3')
            e3 = tf.nn.leaky_relu(e3)

            e4 = ly.conv2d(e3,512,strides = 2,name = 'g_conv2d_4')# 8 * 8 * 512
            e4 = ly.bn_layer(e4,name = 'g_bn_4')
            e4 = tf.nn.leaky_relu(e4)

            e5 = ly.conv2d(e4,512,strides = 2,name = 'g_conv2d_5')# 4 * 4 * 512
            e5 = ly.bn_layer(e5,name = 'g_bn_5')
            e5 = tf.nn.leaky_relu(e5)

            e6 = ly.conv2d(e5,512,strides = 2,name = 'g_conv2d_6')# 2 * 2 * 512
            e6 = ly.bn_layer(e6,name = 'g_bn_6')
            e6 = tf.nn.leaky_relu(e6)

            e7 = ly.conv2d(e6,512,strides = 2,name = 'g_conv2d_7')# 1 * 1 * 512
            e7 = ly.bn_layer(e7,name = 'g_bn_7')
            e7 = tf.nn.leaky_relu(e7)

            d1 = ly.deconv2d(e7,512,2,strides = 2,name = 'g_deconv2d_1')# 2 * 2 * 512
            d1 = ly.bn_layer(d1,name = 'g_bn_8')
            d1 = tf.nn.dropout(tf.nn.leaky_relu(d1),0.5)
            d1 = tf.concat([d1,e6],3)

            d2 = ly.deconv2d(d1,512,4,strides = 2,name = 'g_deconv2d_2')# 4 * 4 * 512
            d2 = ly.bn_layer(d2,name = 'g_bn_9')
            d2 = tf.nn.dropout(tf.nn.leaky_relu(d2),0.5)
            d2 = tf.concat([d2,e5],3)

            d3 = ly.deconv2d(d2,512,8,strides = 2,name = 'g_deconv2d_3')# 8 * 8 * 512
            d3 = ly.bn_layer(d3,name = 'g_bn_10')
            d3 = tf.nn.dropout(tf.nn.leaky_relu(d3),0.5)
            d3 = tf.concat([d3,e4],3)

            d4 = ly.deconv2d(d3,512,16,strides = 2,name = 'g_deconv2d_4')# 16 * 16 * 512
            d4 = ly.bn_layer(d4,name = 'g_bn_11')
            d4 = tf.nn.leaky_relu(d4)
            d4 = tf.concat([d4,e3],3)

            d5 = ly.deconv2d(d4,256,32,strides = 2,name = 'g_deconv2d_5')# 32 * 32 * 256
            d5 = ly.bn_layer(d5,name = 'g_bn_12')
            d5 = tf.nn.leaky_relu(d5)
            d5 = tf.concat([d5,e2],3)

            d6 = ly.deconv2d(d5,128,64,strides = 2,name = 'g_deconv2d_6')# 64 * 64 * 128
            d6 = ly.bn_layer(d6,name = 'g_bn_13')
            d6 = tf.nn.leaky_relu(d6)
            d6 = tf.concat([d6,e1],3)

            d7 = ly.deconv2d(d6,64,128,strides = 2,name = 'g_deconv2d_7')# 128 * 128 * 64
            d7 = ly.bn_layer(d7,name = 'g_bn_14')
            d7 = tf.nn.leaky_relu(d7)
            d7 = tf.concat([d7,e0],3)

            d7 = ly.deconv2d(d7,3,256,strides = 2,name = 'g_deconv2d_8')# 256 * 256 * 3
            d7 = tf.nn.tanh(d7)

        return d7

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.name)

class discriminator(object):
    def __init__(self,name):
        self.name = name
        self.reuse = tf.AUTO_REUSE

    def __call__(self,input):
        with tf.variable_scope(self.name ,reuse = self.reuse):
            input = ly.conv2d(input,64,strides = 2,name = 'd_conv2d_0')
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input,128,strides = 2,name = 'd_conv2d_1')
            input = ly.bn_layer(input,name = 'd_bn_1')
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input,256,strides = 2,name = 'd_conv2d_2')
            input = ly.bn_layer(input,name = 'd_bn_2')
            input = tf.nn.leaky_relu(input)

            input = ly.conv2d(input,512,strides = 2,name = 'd_conv2d_3')
            input = ly.bn_layer(input,name = 'd_bn_3')
            input = tf.nn.leaky_relu(input)

            input = ly.fc(input,1,bias = True,name = 'd_fc_0')

            return tf.nn.sigmoid(input)
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.name)



class pix2pix(op_base):
    def __init__(self,sess,args,reader):
        op_base.__init__(self,args)
        self.sess = sess
        self.Reader = reader(args)

    def build(self):
        self.input_image = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_image_height,self.input_image_weight,self.input_image_channels])
        self.real_image = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_image_height,self.input_image_weight,self.input_image_channels])

        self.G = generator('G')
        self.D = discriminator('D')

        self.fake = self.G(self.input_image)

        self.d_fake = self.D(tf.concat([self.input_image,self.fake], axis = 3))
        self.d_real = self.D(tf.concat([self.input_image,self.real_image], axis = 3))

        safe_log = 1e-12
        self.d_loss = - tf.reduce_mean( tf.log( 1 - self.d_fake + safe_log ) ) - tf.reduce_mean( tf.log( 1 - self.d_fake + safe_log ) )
        self.g_loss_log = - tf.reduce_mean( tf.log(self.d_fake + safe_log) )
        self.g_loss_l1 = tf.reduce_mean(tf.abs( self.fake - self.real_image ))
        self.g_loss = self.g_loss_log + self.g_loss_l1


        self.d_loss_summary = tf.summary.scalar('d_loss',self.d_loss)
        self.g_loss_log_summary = tf.summary.scalar('g_loss_log',self.g_loss_log)
        self.g_loss_l1_summary = tf.summary.scalar('g_loss_l1',self.g_loss_l1)

        self.d_opt = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss,var_list = self.D.vars)
        self.g_opt = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss,var_list = self.G.vars)

        # with tf.control_dependencies([self.d_opt,self.g_opt]):
        #     return tf.no_op(name = 'optimizer')

    def train(self,need_train = True,pretrain = False):

        self.build()
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.sess.run(tf.global_variables_initializer())

        if(pretrain):
            self.saver.restore( self.sess,tf.train.latest_checkpoint(self.model_save_path) )
            print('sucess restore %s' % tf.train.latest_checkpoint(self.model_save_path) )

        if(need_train):
            self.d_sum = tf.summary.merge([self.d_loss_summary])
            self.g_sum = tf.summary.merge([self.g_loss_log_summary,self.g_loss_l1_summary])
            self.writer = tf.summary.FileWriter('./logs',self.sess.graph)

            data_path = os.path.join('data',self.model, self.data_name, 'train')
            data_list = os.listdir(data_path)

            counter = 1

            for i in range(self.epoch):
                epoch_size = len(data_list)

                for batch_time in tqdm(range(epoch_size // self.batch_size)):
                    image_a,image_b = self.Reader.build_batch(batch_time, data_list, data_path)

                    ### first time d
                    _,d_loss,d_sum_str = self.sess.run([self.d_opt,self.d_loss,self.d_sum],feed_dict = { self.input_image: image_b, self.real_image:image_a })
                    self.writer.add_summary(d_sum_str,counter)

                    ### first time g
                    _,g_loss_log,g_loss_l1,g_sum_str = self.sess.run([self.g_opt,self.g_loss_log,self.g_loss_l1,self.g_sum],feed_dict = { self.input_image: image_b, self.real_image:image_a })
                    self.writer.add_summary(g_sum_str,counter)

                    ### second time g
                    _,g_sum_str = self.sess.run([self.g_opt,self.g_sum],feed_dict = { self.input_image: image_b, self.real_image:image_a })
                    self.writer.add_summary(g_sum_str, counter)

                    counter += 1

                    if(counter % 100 == 0):
                        self.saver.save(self.sess,os.path.join(self.model_save_path, 'checkpoint' + '-' + str(i) + '-' + str(batch_time)))

    def test(self):
        self.train(need_train = False,pretrain = True)

        data_path = os.path.join('data', self.model, self.data_name, 'val')
        data_list = os.listdir(data_path)

        image_a,image_b =  self.Reader.build_batch(self, batch_time, data_list, data_path)
        fake = self.sess.run(self.fake,feed_dict = {self.input_image: image_b, self.real_image: image_a })

        write_shape = [self.output_image_height, self.output_image_weight, self.input_image_channels]
        write_image(fake,self.generate_image_path,write_shape)






















