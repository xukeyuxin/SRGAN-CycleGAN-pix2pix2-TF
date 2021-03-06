
import os

class op_base(object):
    def __init__(self,args):

        self.epoch = args.epoch
        self.model = args.model
        self.data_name = args.data_name

        self.batch_size = args.batch_size
        self.input_size = args.input_size

        self.input_image_weight = args.input_image_weight
        self.input_image_height = args.input_image_height
        self.output_image_height = args.output_image_height
        self.output_image_weight = args.output_image_weight
        self.input_image_channels = args.input_image_channels

        if(args.model == 'CGAN'):
            self.label_embedding_size = args.label_embedding_size
            self.input_noise_size = args.input_noise_size
        if(args.model == 'SRGAN'):
            self.init_g_epoch = args.init_g_epoch
            self.model_init_g_save_path = os.path.join('model',self.model,'init_g_prarms')
            if (not os.path.exists(self.model_init_g_save_path)):
                if (not os.path.exists(os.path.join('model', self.model))):
                    os.mkdir(os.path.join('model', self.model))
                os.mkdir(self.model_init_g_save_path)


        self.alpha = 1
        self.beta = 1
        self.lr = args.lr
        self.pretrain = args.pretrain
        self.real_label = args.real_label
        self.max_to_keep = args.max_to_keep

        self.test_type = args.test_type


        self.model_save_path = os.path.join('model',self.model,self.data_name)
        if ( not os.path.exists(self.model_save_path) ):
            if(not os.path.exists(os.path.join('model',self.model))):
                os.mkdir(os.path.join('model',self.model))
            os.mkdir(self.model_save_path)



        self.generate_image_path = os.path.join('data',self.model,self.data_name + '_generate_image')
        if ( not os.path.exists(self.generate_image_path) ):
            if(not os.path.exists(os.path.join('data',self.model))):
                os.mkdir(os.path.join('data',self.model))
            os.mkdir(self.generate_image_path)