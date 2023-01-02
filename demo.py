import torch
import numpy as np
import random
import os
import argparse

from loader.sampler import LabelSampler
from loader.data_loader import load_data_for_MultiDA
from model.model import RRL
from utils.optimizer import get_optimizer
from utils.train_multi import finetune_for_mutil_UDA_sep,train_for_Multi
from utils.eval import predict
from utils import globalvar as gl
# import dataloader as dir_dataloader


import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='officehome',
                    help='the name of dataset')
parser.add_argument('--source', type=str, nargs='+', default=['Art', 'Clipart', 'Real'],
                    help='source domain')
parser.add_argument('--target', type=str, default='Product',
                    help='target domain')
parser.add_argument('--net', type=str, default='resnet',
                    choices=['alexnet', 'vgg16', 'resnet34', 'resnet', 'resnet18', 'resnet101'],
                    help='which network to use')
parser.add_argument('--phase', type=str, default='train',
                    choices=['pretrain', 'train'],
                    help='the phase of training model')
parser.add_argument('--root_dir', type=str, default='./data',
                    help='root dir of the dataset')
parser.add_argument('--gpu', type=int, default=0,
                    help='gpu')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr_mult', type=float, nargs=4, default=[0.1, 0.1, 1, 1],
                    help='lr_mult (default: [0.1, 0.1, 1, 1])')
parser.add_argument('--presteep', type=int, default=20000,
                    help='number of presteep to pretrain (default: 20000)')
parser.add_argument('--steps', type=int, default=200000,
                    help='maximum number of iterations to train (default: 200000)')
parser.add_argument('--lam_step', type=int, default=20000,
                    help='factor of lamda (default: 20000)')
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status(default: 100)')
parser.add_argument('--save_interval', type=int, default=500,
                    help='how many batches to wait before saving a model(default: 500)')
parser.add_argument('--update_interval', type=int, default=1000,
                    help='how many batches to wait before updating pseudo labels(default: 1000)')
parser.add_argument('--start_update_step', type=int, default=2000,
                    help='how many batches to wait before the first time to update the pseudo labels(default: 2000)')
parser.add_argument('--save_check', type=bool, default=True,
                    help='save checkpoint or not(default: True)')
parser.add_argument('--patience', type=int, default=10,
                    help='early stopping to wait for improvment before terminating. (default: 12 (6000 iterations))')
parser.add_argument('--early', type=bool, default=True,
                    help='early stopping or not(default: True)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='MOMENTUM of SGD (default: 0.9)')
parser.add_argument('--decay', type=int, default=0.0005,
                    help='DECAY of SGD (default: 0.0005)')
parser.add_argument('--gamma', type=float, default=1.0  ,
                    help='the trade off factor')     
parser.add_argument('--message', type=str, default='Riemannian Representation Learning',
                    help='the annotation')   

parser.add_argument("--pre_type", default=2,choices=[1,2], type=int, help="type of pre-processing, 1:normal, 2:jitter")


parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
parser.add_argument("--flip", default=0.5, type=float, help="Chance of random horizontal flip")
parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")

# ===========================================================================================================================================================================
args = parser.parse_args()


DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

gl._init()
gl.set_value('DEVICE', DEVICE)


domain_name = {
    'officehome':  ['Art', 'Clipart', 'Real', 'Product'],
}[args.dataset]

class_num = {

    'officehome':65,

}[args.dataset]

bottleneck_dim = {

    "resnet": 1024
}[args.net]

# batch_size of src_l and tar_ul are both batch_size
batch_size = {

    "resnet": 32
}[args.net]

select_class_num = batch_size // 2 # init value of (batch_size//select_class_num) of src_l and tar_ul are 1 and 2 respectively
while select_class_num > class_num:
    select_class_num //= 2




print(args)

seed = 10
torch.manual_seed(seed)
random.seed(seed)

if args.target in args.source:
    raise Exception('The target must not be in source list!')
    quit()
if args.target not in domain_name:
    raise Exception('The target domain: %s NOT in dataset!' % args.target)
    quit()
for name in args.source:
    if name not in domain_name:
        raise Exception('The source domain: %s NOT in dataset!' % name)
        quit()

args.source.sort()

record_dir = 'record_Multi/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
check_path = 'save_model_Multi/{}/{}'.format(str.upper(args.dataset), str.upper(args.net))
if not os.path.exists(check_path):
    os.makedirs(check_path)


record_file = os.path.join(record_dir, 'Multi_{}_{}_to_{}.txt'.format(args.net, args.source, args.target))

gl.set_value('check_path', check_path)
gl.set_value('record_file', record_file)

if __name__ == '__main__':

    label = 0
    domain_labels = [] 
    for ele in args.source:
        if ele:
            domain_labels.append(label)
            label+=1 
    domain_labels.append(label) 
    dataloaders = {}
    model = RRL(args.net, class_num, bottleneck_dim, domain_labels).to(DEVICE)

    with open(gl.get_value('record_file'), 'a') as f: 
        if args.message:
            f.write('\n' + args.message + '\n')
        f.write(" the gamma in the loss is %f \n"%args.gamma)
        f.write(" the init leraning rate is %f \n"%args.lr)
        
    if args.phase == 'pretrain':
        dataloaders['src_pretrain'], dataloaders['tar_test'] = load_data_for_MultiDA(
            args, args.root_dir, args.dataset, args.source, args.target, args.phase, batch_size, args.net)

        for ele in dataloaders['src_pretrain']:
            print(len(ele.dataset))
        print(len(dataloaders['tar_test'].dataset))
        optimizer = get_optimizer(model, args.lr, args.lr_mult)

        finetune_for_mutil_UDA_sep(args, model, optimizer, dataloaders)

    if args.phase == 'train':
        model_path = '{}/best(PT)_{}_{}.pth'.format(check_path, args.net, args.source)
        print('model_path:{}'.format(model_path))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

        if args.start_update_step <= args.update_interval:
            first_batch_num = args.update_interval
        else:
            a = args.start_update_step % args.update_interval
            first_batch_num = args.start_update_step if a == 0  else (args.start_update_step // args.update_interval + 1) * args.update_interval
        label_sampler = LabelSampler(first_batch_num * len(domain_labels), class_num, select_class_num) 
        dataloaders['src_train_l_list'], dataloaders['tar_train_ul'] = load_data_for_MultiDA(
            args, args.root_dir, args.dataset, args.source, args.target, args.phase, batch_size, args.net, label_sampler, pre_type=args.pre_type)
        _, dataloaders['tar_test'] = load_data_for_MultiDA(
            args, args.root_dir, args.dataset, args.source, args.target, 'test', batch_size, args.net, pre_type=args.pre_type)
        print("Source datasets sizes:")
        for ele in dataloaders['src_train_l_list']:
            print(len(ele.dataset))
        print("Target datasets sizes:")
        print(len(dataloaders['tar_train_ul'].dataset))

        # The shuffling and drop_last setting of the target test dataloader is false, so the sequence is the same as the target dataset.
        pseudo_labels = predict(model, dataloaders['tar_test'])
        dataloaders['tar_train_ul'].dataset.update_pseudo_labels(pseudo_labels)

        optimizer = get_optimizer(model, args.lr, args.lr_mult, args.momentum, args.decay)

        train_for_Multi(args, model, optimizer, dataloaders, domain_labels)

