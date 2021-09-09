# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import argparse
import os
import random

import numpy as np
import torch

from ofa.imagenet_classification.elastic_nn.modules.dynamic_op import \
    DynamicSeparableConv2d
from ofa.imagenet_classification.elastic_nn.networks import (OFAMobileNetV3,
                                                             OFAResNets)
from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import \
    load_models
from ofa.imagenet_classification.networks import MobileNetV3Large
from ofa.imagenet_classification.run_manager import \
    DistributedImageNetRunConfig, ImagenetRunConfig, CifarRunConfig
from ofa.imagenet_classification.run_manager.run_manager import (
    RunManager)
from ofa.utils import MyRandomResizedCrop, download_url

# imagenet tranformer + cifar dataset = top1 83%

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='max', choices=[
    'kernel', 'depth', 'expand',
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()
'''Task
task = Kernel : kernel size search
task = Depth  : model depth search
task = expand : kernel depth -> kernel depth width 
'''
if args.task == 'kernel':
    args.path = 'exp/normal2kernel'
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 3e-2
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = '3,5'  # 原3 5 7
    args.expand_list = '6'
    args.depth_list = '4'
elif args.task == 'depth':
    args.path = 'exp/kernel2kernel_depth/phase%d' % args.phase
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = '3,5'
        args.expand_list = '2,4'
        args.depth_list = '2,3'
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5'
        args.expand_list = '2,4'
        args.depth_list = '2,3'
elif args.task == 'expand':
    args.path = 'exp/kernel_depth2kernel_depth_width/phase%d' % args.phase
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 2.5e-3
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.ks_list = '3,5'
        args.expand_list = '2,4'
        args.depth_list = '2,3'
    else:
        args.n_epochs = 120
        args.base_lr = 7.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5'
        args.expand_list = '2,4'
        args.depth_list = '2,3'
elif args.task == "max":
    args.path = 'exp/teachernet'
    args.dynamic_batch_size = 1
    args.n_epochs = 200
    args.base_lr = 0.025
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = '5'
    args.expand_list = '4'
    args.depth_list = '3'
else:
    raise NotImplementedError
args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 128
args.valid_size = 10000

args.opt_type = 'sgd'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-4 #3e-5 
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False

args.model_init = 'he_fout'
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = 'tf'
args.image_size = '32'
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
args.kd_type = 'ce'


if __name__ == '__main__':
    os.makedirs(args.path, exist_ok=True)

    num_gpus = 0

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size)
                       for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]

    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr   # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr

    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4

    run_config = CifarRunConfig(
        **args.__dict__, num_replicas=num_gpus, rank=0)

    # print run config information

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [float(
        width_mult) for width_mult in args.width_mult_list.split(',')]  # 类似slimmable network
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]

    args.width_mult_list = args.width_mult_list[0] if len(
        args.width_mult_list) == 1 else args.width_mult_list

    # teacher model
    args.teacher_model = MobileNetV3Large(
        n_classes=run_config.data_provider.n_classes, bn_param=(
            args.bn_momentum, args.bn_eps),
        dropout_rate=0, width_mult=1.0, ks=7, expand_ratio=6, depth_param=4,
    )
    args.teacher_model.cuda()

    """ RunManager """
    run_manager = RunManager(args.path, args.teacher_model, run_config)

    run_manager.save_config()

    # training
    from ofa.imagenet_classification.elastic_nn.training.progressive_shrinking import (
        train_max, validate_max)

    validate_func_dict = {'image_size_list': {32} if isinstance(args.image_size, int) else sorted({24, 32}),
                          'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
                          'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                          }

    train_max(run_manager, args,
            lambda _run_manager, epoch, is_test: validate_max(_run_manager, epoch, is_test, **validate_func_dict))
    