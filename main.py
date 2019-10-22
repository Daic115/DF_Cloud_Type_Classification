#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   main_cls.py
@Desciption     :   None
@Modify Time      @Author    @Version
------------      -------    --------
2019/10/6 1:25   Daic       1.0
'''
import os
import json
import argparse
from untils import *
from trainer import *

parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--input_json', type=str, default='./data/dataset_k1.json')
parser.add_argument('--image_path', type=str, default='./data/train')#'./data/train')
#output_path
parser.add_argument('--out_path', type=str,  default='./data/save')
parser.add_argument('--train_name', type=str,  default='resnet50')
#dataset
parser.add_argument('--data_type', type=str,  default='kfolder',
                    help='kfolder: using K folder split ; train_all: train all images with out validation ')
parser.add_argument('--size', type=int,  default=224)
parser.add_argument('--precrop', type=bool,  default=True,
                    help='crop image which is too high or wide')
#optimizer
##more detail can be found in untils.build_optimizer function
parser.add_argument('--optimizer', type=str,  default='sgdmom')
parser.add_argument('--lr', type=float,  default=3e-4)
parser.add_argument('--weight_decay', type=float,  default=1e-5)
parser.add_argument('--momentum', type=float,  default=0.9)
parser.add_argument('--split_weights', type=bool,  default=True,
                    help='no weight decay for bias: from bag of tricks')

##you can define your lr schedule in traner.py
##if use epoch schedule
parser.add_argument('--decay_step', type=list,  default=[4,8,12])
parser.add_argument('--decay_rate', type=float,  default=0.6)

#model
parser.add_argument('--cnn', type=str,  default='resnet50')
parser.add_argument('--num_class', type=int,  default=29)
parser.add_argument('--loss', type=str,  default='lsr',
                    help='lsr:      label smooth ce with default 0.1 smooth factor'
                         'ce :      Cross Entropy Loss'
                         '!Carefully use:'
                         'bce:      nn.BCELoss()'
                         'bcel:     nn.BCEWithLogitsLoss()'
                         'focal:    Focal Loss')

#train and eval
parser.add_argument('--train_bch', type=int,  default=64)
parser.add_argument('--val_bch', type=int,  default=64)
parser.add_argument('--num_worker', type=int,  default=8)
parser.add_argument('--max_epoch', type=int,  default=20)

parser.add_argument('--seed', type=int,  default=1996)
parser.add_argument('--gpu_id', type=str,  default='2')

#resume
parser.add_argument('--resume_path', type=str,  default='')

def train():
    opt = parser.parse_args()
    dopt = vars(opt)
    #make save path and save config:
    save_path = os.path.join(opt.out_path, opt.train_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    opt_save_path = os.path.join(opt.out_path,opt.train_name,'OPTIONS_'+opt.cnn+'.json')
    json.dump(dopt,open(opt_save_path,'w'))
    #fix gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    #fix seed
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.
    np.random.seed(opt.seed)  # Numpy module.
    os.environ["PYTHONHASHSEED"] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True


    trainer = MultiClsTrainer(opt)
    #trainer.lr_finder()
    trainer.train_model()

def load_options(options,dict):
    for k in dict:
        vars(options)[k] = dict[k]
    return options

def test(option_path,
         test_image_folder,
         save_path,
         model_path,
         test_info_path = './data/dataset_test.json'):

    opt = parser.parse_args()
    opt = load_options(opt,json.load(open(option_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    trainer = MultiClsTrainer(opt)
    trainer.test_model(test_info_path,test_image_folder,save_path,model_path)

