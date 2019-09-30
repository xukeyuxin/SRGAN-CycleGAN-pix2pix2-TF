# coding=utf-8
import tensorflow as tf
import numpy as np
import os
from functools import reduce


def get_variable(name, shape, initializer, scope):
    try:
        v = tf.get_variable(name, shape=shape, initializer=initializer)
    except ValueError:
        scope.reuse_variables()
        v = tf.get_variable(name, shape=shape, initializer=initializer)
    return v


def conv2d(input, output_shape, strides=1, kernel_size=3, padding='SAME', name=None, bias=False ):
    input_shape = input.get_shape().as_list()[-1]
    filter_shape = [kernel_size, kernel_size, input_shape, output_shape]
    strides_shape = [1, strides, strides, 1]
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape=filter_shape, initializer=initializer)
        conv = tf.nn.conv2d(input, filter=weight, strides=strides_shape, padding=padding)
        if (bias):
            _bias = tf.get_variable('bias', shape = filter_shape[-1], initializer = tf.constant_initializer(0.))
            conv = tf.nn.bias_add(conv, _bias)
    return conv



def deconv2d(input, output_channel, output_size, strides=1, kernel_size=3,padding='SAME', name=None, bias=False):
    input_channel = input.get_shape().as_list()[-1]
    batch_shape = input.get_shape().as_list()[0]
    filter_shape = [kernel_size, kernel_size, output_channel, input_channel]
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
    strides_shape = [1, strides, strides, 1]
    output_shape = [batch_shape, output_size, output_size, output_channel]

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape = filter_shape, initializer = initializer)
        conv = tf.nn.conv2d_transpose(input, filter=weight, output_shape=output_shape, strides=strides_shape,
                                      padding=padding)
        if(bias):
            _bias = tf.get_variable('bias', shape = filter_shape[-1], initializer = tf.constant_initializer(0.))
            conv = tf.nn.bias_add(conv, _bias)

    return conv


def dilated_conv2d(input, output_shape, rate=3, kernel_size=3, padding='SAME',name=None, bias=False):
    input_shape = input.get_shape().as_list()[-1]
    filter_shape = [kernel_size, kernel_size, input_shape, output_shape]
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape = filter_shape, initializer = initializer)
        conv = tf.nn.atrous_conv2d(input, filter=weight, rate=rate, padding=padding)
        if (bias):
            _bias = tf.get_variable('bias', shape = filter_shape[-1], initializer = tf.constant_initializer(0.))
            conv = tf.nn.bias_add(conv, _bias)

    return conv


def fc(input, output_shape, name=None, bias=False,initializer = tf.random_normal_initializer(mean=0., stddev=0.02)):
    input_shape = input.get_shape().as_list()
    input = tf.reshape(input, [-1, reduce(lambda x, y: x * y, input_shape[1:])])

    fc_weight_shape = [reduce(lambda x, y: x * y, input_shape[1:]), output_shape]
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', shape = fc_weight_shape, initializer = initializer)
        fc = tf.matmul(input, weight)
        # fc = tf.layers.dense(input,weight,)
        if (bias):
            _bias = tf.get_variable('bias', shape = fc_weight_shape[-1], initializer = tf.constant_initializer(0.))
            fc = tf.nn.bias_add(fc, _bias)

    return fc

def Upsampling(input,out_size):
    assert len(input.get_shape.as_list()) == 4
    return tf.image.resize_images(input,[out_size,out_size])


def max_pool(input, ksize, strides_size=1, padding='SAME', name=None):
    with tf.variable_scope(name):
        strides = [1, strides_size, strides_size, 1]
        filters = [1, ksize, ksize, 1]
        max_pool = tf.nn.max_pool(input, filters, strides=strides, padding=padding)

    return max_pool


def avg_pool(input, ksize, strides_size=1, padding='SAME', name=None):
    with tf.variable_scope(name):
        strides = [1, strides_size, strides_size, 1]
        filters = [1, ksize, ksize, 1]
        arg_pool = tf.nn.arg_pool(input, filters, strides=strides, padding=padding)

    return arg_pool


def leaky_relu(input):
    return tf.nn.leaky_relu(input)


def l1_loss(x1, x2):
    return tf.reduce_mean(tf.abs(x1 - x2))


def resize_nn(x, size):
    return tf.image.resize_nearest_neighbor(x, size=(int(size), int(size)))


def xavier_init(random_size, name, init_type='normal', trainable=True):
    if (init_type == 'zeros'):
        initializer = tf.zeros(shape=random_size)
    else:
        mean = 0
        stddev = 1 / tf.sqrt(random_size[0] / 2.0)
        initializer = tf.random_normal(shape=random_size, mean=mean, stddev=stddev)
    return tf.Variable(initializer, name=name, trainable=trainable)



def bn_layer(x, is_training=True, name=None, moving_decay=0.9, eps=1e-5,gamma_initializer = tf.constant_initializer(1)):
    # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
    shape = x.shape
    assert len(shape) in [2, 4]

    param_shape = shape[-1]
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta

        # 计算当前整个batch的均值与方差
        axes = list(range(len(shape) - 1))
        batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')

        # 采用滑动平均更新均值与方差
        ema = tf.train.ExponentialMovingAverage(moving_decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
        mean, var = tf.cond(tf.equal(is_training, True), mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        gamma = tf.get_variable('gamma', shape = param_shape, initializer = gamma_initializer)
        beta = tf.get_variable('beta', shape = param_shape, initializer = tf.constant_initializer(0))

        # 最后执行batch normalization
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

class vgg19(object):
    def __init__(self,name):
        self.reuse = False
        self.name = name
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3','conv3_4',
                       'conv4_1','conv4_2','conv4_3','conv4_4','conv5_1','conv5_2','conv5_3','conv5_4',
                       'fc6','fc7','fc8']
        vgg_path = os.path.join('model','VGG19','vgg19-full.npy')
        self.npz = np.load(vgg_path,encoding = 'latin1', allow_pickle = True).item()
        # for i in sorted(self.npz.items()):
        #     print('layer: %s  weight %s bias %s' % (i[faces],i[1][faces].shape,i[1][1].shape))

    def _conv2d(self,input,filter,name):
        with tf.variable_scope(name):
            filter = self.npz[name][0]
            bias = self.npz[name][1]
            conv = tf.nn.conv2d(input,filter,[1,1,1,1],padding = 'SAME')
            conv = tf.nn.bias_add(conv,bias)
            conv = tf.nn.relu(conv)
        return conv

    def _fc(self,input,name):
        with tf.variable_scope(name):
            weight = self.npz[name][0]
            bias = self.npz[name][1]
            fc = tf.nn.bias_add(tf.matmul(input,filter),bias)
        return fc

    def _maxpool(self,input,name):
        return tf.nn.max_pool(input,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME',name = name)

    def prepare(self,input):
        ### [-1,1] -> [faces,255]
        input = tf.image.resize_images(input,(224,224))
        input = 255. * (input + 1)
        red,green,blue = tf.split(input,3,axis = 3)
        red = red - self.VGG_MEAN[0]
        green = green - self.VGG_MEAN[1]
        blue = blue - self.VGG_MEAN[2]
        input = tf.concat([red,green,blue],axis = 3)
        return input


    def __call__(self,input,need_fc = False):
        input = self.prepare(input)
        with tf.variable_scope(self.name,reuse = self.reuse):
            self.conv1_1 = self._conv2d(input,64,name = 'conv1_1')
            self.conv1_2 = self._conv2d(self.conv1_1,64,name = 'conv1_2')
            self.maxpool1 = self._maxpool(self.conv1_2,name = 'maxpool1')

            self.conv2_1 = self._conv2d(self.maxpool1,128,name = 'conv2_1')
            self.conv2_2 = self._conv2d(self.conv2_1,128,name = 'conv2_2')
            self.maxpool2 = self._maxpool(self.conv2_2, name='maxpool2')

            self.conv3_1 = self._conv2d(self.maxpool2,256,name = 'conv3_1')
            self.conv3_2 = self._conv2d(self.conv3_1,256,name = 'conv3_2')
            self.conv3_3 = self._conv2d(self.conv3_2,256,name = 'conv3_3')
            self.conv3_4 = self._conv2d(self.conv3_3,256,name = 'conv3_4')
            self.maxpool3 = self._maxpool(self.conv3_4, name='maxpool3')

            self.conv4_1 = self._conv2d(self.maxpool3,512,name = 'conv4_1')
            self.conv4_2 = self._conv2d(self.conv4_1,512,name = 'conv4_2')
            self.conv4_3 = self._conv2d(self.conv4_2,512,name = 'conv4_3')
            self.conv4_4 = self._conv2d(self.conv4_3,512,name = 'conv4_4')
            self.maxpool4 = self._maxpool(self.conv4_4, name='maxpool4')

            self.conv5_1 = self._conv2d(self.maxpool4,512,name = 'conv5_1')
            self.conv5_2 = self._conv2d(self.conv5_1,512,name = 'conv5_2')
            self.conv5_3 = self._conv2d(self.conv5_2,512,name = 'conv5_3')
            self.conv5_4 = self._conv2d(self.conv5_3,512,name = 'conv5_4')
            self.maxpool5 = self._maxpool(self.conv5_4, name='maxpool5')

            if(not need_fc):
                return self.maxpool5

            self.maxpool5 = tf.reshape(self.maxpool5,[-1,7*7*512])
            self.fc6 = self._fc(self.maxpool5,'fc6') ## 25088 -> 4096
            self.fc6 = tf.nn.relu(self.fc6)
            self.fc7 = self._fc(self.fc6,'fc7') ## 4096 -> 4096
            self.fc7 = tf.nn.relu(self.fc7)
            self.fc8 = self._fc(self.fc7,'fc8') ## 4096 -> 1000
            self.fc8 = tf.nn.softmax(self.fc8)

            return self.fc8

class vgg16(object):
    def __init__(self,name):
        self.reuse = False
        self.name = name
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.layers = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3',
                       'conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3','fc6','fc7','fc8']
        vgg_path = os.path.join('model','VGG16','vgg16.npy')
        self.npz = np.load(vgg_path,encoding = 'latin1', allow_pickle = True).item()
        # for i in sorted(self.npz.items()):
        #     print('layer: %s  weight %s bias %s' % (i[faces],i[1][faces].shape,i[1][1].shape))

    def _conv2d(self,input,name):
        with tf.variable_scope(name):
            filter = self.npz[name][0]
            bias = self.npz[name][1]
            conv = tf.nn.conv2d(input,filter,[1,1,1,1],padding = 'SAME')
            conv = tf.nn.bias_add(conv,bias)
            conv = tf.nn.relu(conv)
        return conv

    def _fc(self,input,name):
        with tf.variable_scope(name):
            weight = self.npz[name][0]
            bias = self.npz[name][1]
            fc = tf.nn.bias_add(tf.matmul(input,filter),bias)
        return fc

    def _maxpool(self,input,name):
        return tf.nn.maxpool2d(input,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME',name = name)

    def __call__(self,input,need_fc = False):
        with tf.variable_scope(self.name,reuse = self.reuse):
            self.conv1_1 = self._conv2d(input,64,name = 'conv1_1')
            self.conv1_2 = self._conv2d(self.conv1_1,64,name = 'conv1_2')
            self.maxpool1 = self._maxpool(self.conv1_2,name = 'maxpool1')

            self.conv2_1 = self._conv2d(maxpool1,128,name = 'conv2_1')
            self.conv2_2 = self._conv2d(conv2_1,128,name = 'conv2_2')
            self.maxpool2 = self._maxpool(self.conv2_2, name='maxpool2')

            self.conv3_1 = self._conv2d(maxpool2,256,name = 'conv3_1')
            self.conv3_2 = self._conv2d(conv3_1,256,name = 'conv3_2')
            self.conv3_3 = self._conv2d(conv3_2,256,name = 'conv3_3')
            self.maxpool3 = self._maxpool(self.conv3_4, name='maxpool3')

            self.conv4_1 = self._conv2d(maxpool3,256,name = 'conv4_1')
            self.conv4_2 = self._conv2d(conv4_1,256,name = 'conv4_2')
            self.conv4_3 = self._conv2d(conv4_2,256,name = 'conv4_3')
            self.maxpool4 = self._maxpool(self.conv4_4, name='maxpool4')

            self.conv5_1 = self._conv2d(maxpool2,256,name = 'conv5_1')
            self.conv5_2 = self._conv2d(conv5_1,256,name = 'conv5_2')
            self.conv5_3 = self._conv2d(conv5_2,256,name = 'conv5_3')
            self.maxpool5 = self._maxpool(self.conv5_4, name='maxpool5')

            self.fc6 = self._fc(self.maxpool5,'fc6') ## 25088 -> 4096
            self.fc6 = tf.nn.relu(self.fc6)
            self.fc7 = self._fc(self.fc6,'fc7') ## 4096 -> 4096
            self.fc7 = tf.nn.relu(self.fc7)
            self.fc8 = self._fc(self.fc7,'fc8') ## 4096 -> 1000
            self.fc8 = tf.nn.softmax(self.fc8)

            return_layer = self.maxpool5 if not need_fc else self.fc8

            return return_layer



