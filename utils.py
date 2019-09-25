import numpy as np
import os
import cv2
from op_base import op_base
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def build_data():
    data_name = 'apple2orange'
    content_dir = 'trainB'

    new_data_name = 'apple_orange'
    content_path = os.path.join(new_data_name, content_dir)

    count = 0

    if (not os.path.exists(content_path)):
        if (not os.path.exists(new_data_name)):
            os.mkdir(new_data_name)
        os.mkdir(content_path)

    image_dir_path = os.path.join('data',self.model,data_name,content_dir)
    picture_list = os.listdir(image_dir_path)
    count = 0
    for one in picture_list:
        image_path = os.path.join(image_dir_path,one)
        image = cv2.imread(image_path)
        new_image = cv2.resize(image,(128,128),interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(content_path,one),new_image)
        count += 1

    if(count % 10 == 0):
        print('finish count %s' % count)

def rgb2float(input):
    return input / 127.5 - 1

def float2rgb(input):
    return tf.image.convert_image_dtype((input + 1.0) / 2.0, tf.uint8)

    # output = (input + 1) * 127.5
    # output = output.astype(np.int16)

def float2gray(input):
    return tf.image.convert_image_dtype(input,tf.uint8)

def write_image(image_content,path,write_shape):

    image_content = tf.map_fn(float2rgb, image_content, dtype=tf.uint8).eval()
    image_zip_content = zip(range(len(image_content)),image_content)
    for key,one in image_zip_content:
        one = one.reshape(write_shape)
        cv2.imwrite(os.path.join(path,str(key) + '.jpg'), one)

def write_image_gray(image_content,path,write_shape):
    image_content = tf.map_fn(float2gray, image_content, dtype=tf.uint8).eval()
    image_zip_content = zip(range(len(image_content)),image_content)
    for key,one in image_zip_content:
        print(one)
        one = one.reshape(write_shape)
        cv2.imwrite(os.path.join(path,str(key) + '.jpg'), one)

class reader(op_base):

    def __init__(self,args):
        op_base.__init__(self,args)

    def build_batch(self,batch_time,data_list,data_path):

        start_count = batch_time * self.batch_size
        end_count = (batch_time + 1) * self.batch_size
        choice_batch = data_list[start_count:end_count]
        one_batch_shape = (1, self.input_image_height, self.input_image_weight, self.input_image_channels)
        init_array = []

        for one in choice_batch:
            image_content = cv2.imread(os.path.join(data_path,one))
            image_content = rgb2float(image_content)
            image_content = np.reshape(image_content,one_batch_shape)
            init_array.append(image_content)

            # with open(os.path.join(data_path,one),'rb') as f:
            #     image_content = f.read()
            #     print(image_content)
            #     standard_shape = (self.input_image_height, self.input_image_weight,self.input_image_channels)
            #     image_content = tf.image.decode_jpeg(image_content, channels=standard_shape[-1])
            #     image_content = tf.image.resize_images(image_content, size=standard_shape[:2])
            #     image_content = rgb2float(image_content)
            #     image_content = tf.reshape(image_content,one_batch_shape)
            #     init_array.append(image_content)

        init_array = np.concatenate(init_array,axis = 0)

        return init_array


class mnist():
	def __init__(self, flag='conv', is_tanh = False):
		datapath = 'data/CGAN/mnist'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 10
		self.size = 28 # for conv
		self.channel = 1 # for conv
		self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh

	def __call__(self,batch_size):
		batch_imgs,y = self.data.train.next_batch(batch_size)
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.size, self.size, self.channel))
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1
		return batch_imgs, y

	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.size,self.size), cmap='Greys_r')
		return fig