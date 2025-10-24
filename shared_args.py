import argparse

def add_shared_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='M',
                        help='eval_mode')  # S: the same to training model, M: multi architectures
    parser.add_argument('--num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--save_it', type=int, default=None, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')

    parser.add_argument('--mom_img', type=float, default=0.5, help='momentum for updating synthetic images')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=128, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='noise', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--space', type=str, default='lfm', choices=['p', 'wp', 'lfm'])
    parser.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='resolution')
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--avg_w', action='store_true')

    parser.add_argument('--eval_all', action='store_true')

    parser.add_argument('--min_it', type=bool, default=False)
    parser.add_argument('--no_aug', type=bool, default=False)

    parser.add_argument('--force_save', action='store_true')

    parser.add_argument('--sg_batch', type=int, default=10)

    parser.add_argument('--rand_f', action='store_true')

    parser.add_argument('--logdir', type=str, default='./logged_files')

    parser.add_argument('--wait_eval', action='store_true')

    parser.add_argument('--idc_factor', type=int, default=1)

    parser.add_argument('--rand_gan_un', action='store_true')
    parser.add_argument('--rand_gan_con', action='store_true')

    parser.add_argument('--learn_g', action='store_true')

    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=5)


    parser.add_argument('--special_gan', default=None)

    parser.add_argument('--model_type', default = 'adm')
    parser.add_argument('--epoch_id', default = 1125)
    parser.add_argument('--exp', default='imnet_f8_adm')
    parser.add_argument('--method', default='dopri5')
    parser.add_argument('--num_steps', default=0)
    parser.add_argument('--cfg', default=1.25)
    parser.add_argument('--img_size', default=32)
    parser.add_argument('--ch_mult',default="1 2 2 2")
    parser.add_argument('--attn_res', default="8 4")
    parser.add_argument('--use_origin_adm', default=True)
    parser.add_argument('--nf', default=128)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--layout', default=False)
    parser.add_argument('--num_in_channels', default = 4)
    parser.add_argument('--num_out_channels', default = 4)
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--num_res_blocks', default =2)
    parser.add_argument('--dropout', default = 0.0)
    parser.add_argument('--resamp_with_conv', default=True)
    parser.add_argument('--num_heads', default = 4)
    parser.add_argument('--num_head_channels', default=-1)
    parser.add_argument('--num_heads_upsample', default=-1)
    parser.add_argument('--use_scale_shift_norm', default=True)
    parser.add_argument('--resblock_updown',default=True)
    parser.add_argument('--use_new_attention_order', default=False)
    parser.add_argument('--label_dropout', default=0.0)



    return parser
