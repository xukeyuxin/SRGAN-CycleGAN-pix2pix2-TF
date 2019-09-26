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
            input = tf.nn.sigmoid(input)

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

class classifier(object):
    def __init__(self,name):
        self.name = name
        self.reuse = tf.AUTO_REUSE
    def __call__(self,input):
        with tf.variable_scope(self.name,reuse = self.reuse):



            input = ly.conv2d(input,64,strides = 2,name = 'conv2d_0') # 14 14 64
            input = ly.bn_layer(input,name = 'bn_0')
            input = tf.nn.relu(input)

            input = ly.conv2d(input,128,strides = 2,name = 'conv2d_1') # 7 7 128
            input = ly.bn_layer(input,name = 'bn_1')
            input = tf.nn.relu(input)


            input = ly.fc(input,1024,name = 'fc_0')
            input = tf.nn.relu(input)

            input = ly.fc(input, 10, name='fc_1')

        return input

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = self.name)



class CGAN(op_base):

    def __init__(self,sess,args,reader):
        op_base.__init__(self,args)
        self.sess = sess
        self.Reader = reader(args)
        self.CGAN_TYPE = 'classifier_w_cgan'

        # data
        self.data = mnist()

    def dc_cgan(self):

        self.x = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_image_weight, self.input_image_height, self.input_image_channels]) ### real image
        self.z = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_noise_size])
        self.y = tf.placeholder(tf.float32,shape = [self.batch_size,self.label_embedding_size])

        # self.g = self.generotor(tf.concat([self.z,self.y],axis = 1))
        self.G = generator('G')
        self.D = discriminator('D')


        self.fake = self.G(tf.concat([self.z,self.y],axis = 1))
        expand_dim_y = tf.reshape(self.y,[self.batch_size,1,1,self.label_embedding_size]) * tf.ones(shape = [self.batch_size,28,28,self.label_embedding_size],dtype = tf.float32 )

        self.d_real = self.D(tf.concat([self.x, expand_dim_y],axis = 3))
        self.d_fake = self.D(tf.concat([self.fake, expand_dim_y],axis = 3))

        safe_log = 1e-12
        ### use cross entropy (safe log)
        # self.g_loss = - tf.reduce_mean( tf.log( tf.nn.sigmoid(self.d_fake) + safe_log) )
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake,labels = tf.ones_like(self.d_fake)) )

        # self.d_loss = - tf.reduce_mean(tf.log( 1- tf.nn.sigmoid(self.d_fake) + safe_log )) - tf.reduce_mean( tf.log(tf.nn.sigmoid(self.d_real) + safe_log ))
        self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake,labels = tf.zeros_like(self.d_fake)) ) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_real,labels = tf.ones_like(self.d_real)) )

        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss,var_list = self.G.vars)
        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss,var_list = self.D.vars)

        with tf.control_dependencies([self.opt_g, self.opt_d]):
            return tf.no_op(name='dc_cgan_optimizer')

    def dc_ls_cgan(self):

        self.x = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_image_weight, self.input_image_height, self.input_image_channels]) ### real image
        self.z = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_noise_size])
        self.y = tf.placeholder(tf.float32,shape = [self.batch_size,self.label_embedding_size])

        self.G = generator('G')
        self.D = discriminator('D')

        self.fake = self.G(tf.concat([self.z,self.y],axis = 1))
        expand_dim_y = tf.reshape(self.y,[self.batch_size,1,1,self.label_embedding_size]) * tf.ones(shape = [self.batch_size,28,28,self.label_embedding_size],dtype = tf.float32 )

        self.d_real = tf.nn.sigmoid(self.D(tf.concat([self.x, expand_dim_y],axis = 3)))
        self.d_fake = tf.nn.sigmoid(self.D(tf.concat([self.fake, expand_dim_y],axis = 3)))

        ### use lsgan
        real_weight = 0.95
        self.g_loss = tf.reduce_mean( tf.square( self.d_fake - real_weight ) )
        self.d_loss = tf.reduce_mean(tf.square(self.d_fake)) + tf.reduce_mean(tf.square(self.d_real-real_weight))

        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss,var_list = self.G.vars)
        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss,var_list = self.D.vars)

        with tf.control_dependencies([self.opt_g,self.opt_d]):
            return tf.no_op(name = 'dc_ls_cgan_optimizer')

    def classifier_w_cgan(self):
        self.x = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_image_weight,self.input_image_height,self.input_image_channels])
        self.z = tf.placeholder(tf.float32,shape = [self.batch_size,self.input_noise_size])
        self.y = tf.placeholder(tf.float32,shape = [self.batch_size,self.label_embedding_size])

        self.G = generator('G')
        self.D = discriminator('D')
        self.C = classifier('C')

        self.fake = self.G(tf.concat([self.z,self.y],axis = 1))

        self.d_real = self.D(self.x)
        self.d_fake = self.D(self.fake)
        self.c_real = self.C(self.x)
        self.c_fake = self.C(self.fake)



        safe_log = 1e-12

        ### g , 最大化 D (self.d_fake) 最大化 分类准确数，log需要负号
        self.g_loss = - tf.reduce_mean( self.d_fake) - tf.reduce_mean( tf.reduce_sum(self.y * tf.log( tf.nn.softmax(self.c_fake) + safe_log) , axis = 1 ) )
        ### d , 最小化 D (self.d_fake) 最大化 D (self.d_real)
        self.d_loss = tf.reduce_mean(self.d_fake) - tf.reduce_mean(self.d_real)

        # self.g_loss = -tf.reduce_mean( tf.log( tf.nn.sigmoid(self.d_fake) + safe_log ) ) - tf.reduce_mean( tf.reduce_sum(self.y * tf.log( tf.nn.softmax(self.c_fake) + safe_log) , axis = 1 ) )
        # self.d_loss =  tf.reduce_mean( tf.log( tf.nn.sigmoid(self.d_fake) + safe_log)) - tf.reduce_mean( tf.log( tf.nn.sigmoid(self.d_real) + safe_log))

        self.clip_D = [var.assign(tf.clip_by_value(var, -0.01,0.01)) for var in self.D.vars]
        self.c_loss = - tf.reduce_mean( tf.reduce_sum(self.y * tf.log( tf.nn.softmax(self.c_real) + safe_log) , axis = 1 ) )

        self.opt_g = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss,var_list = self.G.vars)
        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss,var_list = self.D.vars)
        self.opt_c = tf.train.AdamOptimizer(self.lr).minimize(self.c_loss,var_list = self.C.vars)

        with tf.control_dependencies([self.opt_g,self.opt_d,self.opt_c]):
            return tf.no_op(name = 'classifier_w_cgan_optimizer')


    def z_sample(self):
        return np.random.uniform(-1.,1.,size = [self.batch_size,self.input_noise_size])

    def bulid_optimizer_type_dict(self):
        self.CGAN_OPT_TYPE_DICT = dict()
        self.CGAN_OPT_TYPE_DICT['classifier_w_cgan'] = self.classifier_w_cgan
        self.CGAN_OPT_TYPE_DICT['dc_ls_cgan'] = self.dc_ls_cgan
        self.CGAN_OPT_TYPE_DICT['dc_cgan'] = self.dc_cgan

    def get_optimizer(self,name):
        return self.CGAN_OPT_TYPE_DICT[name]()

    def train(self,need_train = True,pretrain = False):

        self.bulid_optimizer_type_dict()
        optimizer = self.get_optimizer(self.CGAN_TYPE)
        saver = tf.train.Saver()
        self.model_save_path = os.path.join(self.model_save_path,self.CGAN_TYPE)

        if(not os.path.exists(self.model_save_path)):
            os.mkdir(self.model_save_path)

        if(pretrain):
            saver.restore(self.sess,tf.train.latest_checkpoint(self.model_save_path))
        if(not need_train):
            return
        self.sess.run(tf.global_variables_initializer())
        print('start train')
        for num in range(1000000):
            X_b, y_b = self.data(self.batch_size)

            if(self.CGAN_TYPE == 'classifier_w_cgan'):
                _opt,_clip_D,g_loss,d_loss,c_loss = self.sess.run([optimizer,self.clip_D,self.g_loss,self.d_loss,self.c_loss],feed_dict = {self.x:X_b,self.y:y_b,self.z:self.z_sample()})

                if (num % 200 == 0):

                    saver.save(self.sess, os.path.join(self.model_save_path, 'checkpoint' + '-' + str(num)))
                    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_loss: {:.4}'.format(num, d_loss, g_loss,c_loss))

            elif(self.CGAN_TYPE == 'dc_ls_cgan'):
                _,g_loss,d_loss = self.sess.run([optimizer,self.g_loss,self.d_loss],feed_dict={self.x:X_b,self.y:y_b,self.z:self.z_sample()} )

                if (num % 200 == 0):
                    fake = self.sess.run(
                        self.fake,
                        feed_dict={self.x: X_b, self.y: y_b, self.z: self.z_sample()})

                    saver.save(self.sess, os.path.join(self.model_save_path, 'checkpoint' + '-' + str(num)))
                    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(num, d_loss, g_loss))

            elif(self.CGAN_TYPE == 'dc_cgan'):
                _,g_loss,d_loss = self.sess.run([optimizer,self.g_loss,self.d_loss],feed_dict={self.x:X_b,self.y:y_b,self.z:self.z_sample()} )

                if (num % 200 == 0):
                    fake = self.sess.run(
                        self.fake,
                        feed_dict={self.x: X_b, self.y: y_b, self.z: self.z_sample()})

                    saver.save(self.sess, os.path.join(self.model_save_path, 'checkpoint' + '-' + str(num)))
                    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'.format(num, d_loss, g_loss))



    def test(self):
        self.train(need_train = False, pretrain = True)
        X_b, y_b = self.data(self.batch_size)
        fake = self.sess.run(
                self.fake,
                feed_dict={self.x: X_b, self.y: y_b, self.z: self.z_sample()})
        write_shape = [self.input_image_height, self.input_image_weight, self.input_image_channels]
        write_image_gray(fake,self.generate_image_path,write_shape)









