
import os

class op_base(object):
    def __init__(self,args):

        self.epoch = args.epoch
        self.data_name = args.data_name

        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.embedding_size = args.embedding_size
        self.input_noise_size = args.input_noise_size
        self.input_image_weight = args.input_image_weight
        self.input_image_height = args.input_image_height
        self.input_image_channels = args.input_image_channels



        self.alpha = 1
        self.beta = 1
        self.lr = args.lr
        self.pretrain = args.pretrain
        self.real_label = args.real_label

        self.test_type = args.test_type


        self.model_save_path = os.path.join('model',self.data_name)
        if ( not os.path.exists(self.model_save_path) ):
            os.mkdir(self.model_save_path)

        self.generate_image_path = os.path.join('data',self.data_name + '_generate_image')
        if ( not os.path.exists(self.generate_image_path) ):
            os.mkdir(self.generate_image_path)