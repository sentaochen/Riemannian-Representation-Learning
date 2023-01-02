import os
import torch
from torchvision import datasets, transforms

from loader.data_list import ImageList, MutilImageList
from loader.sampler import BatchSampler




def get_train_transformers(args, crop_size):
    img_tr = [transforms.RandomResizedCrop(int(crop_size), (args.min_scale, args.max_scale))]
    if args.flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)

def get_val_transformer(dataset, crop_size, resize_size = 256):
    img_tr = []
    img_tr += [transforms.Resize((crop_size, crop_size))]

    img_tr += [transforms.ToTensor(),
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]
                   
              
    return transforms.Compose(img_tr)

def load_data_for_MultiDA(args, root_dir, dataset, src_domains , tar_domain, phase, batch_size, net, label_sampler=None, is_extract=False, pre_type=1):
    '''
    label_sampler is necessary when phase is 'train'
    '''
    crop_size = 224 if net is not 'alexnet' else 227
    resize_size = 256
    if pre_type==1:
        transform_dict = {
            'train': transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
            ])}
    else:
        transform_dict = {
            'train': get_train_transformers(args, crop_size),
            'val': get_val_transformer(dataset, crop_size),
            'test': get_val_transformer(dataset, crop_size)
            }
    list_root_dir = os.path.join(root_dir, 'list', dataset)
    data_root_dir = os.path.join(root_dir, dataset)
    unlabeled_target_list = os.path.join(list_root_dir, '{}.txt'.format(tar_domain))

    labeled_source_list_array = []
    for src in src_domains:
        if src :
            labeled_source_list_array.append(os.path.join(list_root_dir, '{}.txt'.format(src)))

    if phase=='pretrain':
        src_loader_labeled_list = []
        domain_label = 0
        for labeled_source_list in labeled_source_list_array:
            src_data_labeled = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'], type='labeled')
            src_loader_labeled_list.append(torch.utils.data.DataLoader(src_data_labeled,batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4))
            domain_label += 1
        tar_data_unlabeled = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['test'], type='labeled')
        tar_loader_unlabeled = torch.utils.data.DataLoader(tar_data_unlabeled, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

        return src_loader_labeled_list, tar_loader_unlabeled

    elif phase=='train':
        if label_sampler is None:
            src_loader_labeled_list = []
            domain_label = 0
            for labeled_source_list in labeled_source_list_array:
                src_data_labeled = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'], type='labeled',domain_label=domain_label)
                src_loader_labeled_list.append(torch.utils.data.DataLoader(src_data_labeled, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4))
                domain_label += 1

            tar_data_unlabeled = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['train'], type='unlabeled',domain_label=domain_label)
            tar_loader_unlabeled = torch.utils.data.DataLoader(tar_data_unlabeled, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

            return src_loader_labeled_list, tar_loader_unlabeled

        else:
            loader_labeled_list = []
            domain_label = 0
            for labeled_source_list in labeled_source_list_array:
                src_data_labeled = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'], type='labeled',domain_label=domain_label)
                src_batch_sampler_labeled = BatchSampler(src_data_labeled, label_sampler, batch_size )
                loader_labeled_list.append(torch.utils.data.DataLoader(src_data_labeled, batch_sampler = src_batch_sampler_labeled, num_workers=4))
                domain_label += 1

            tar_data_unlabeled = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['train'], type='unlabeled', domain_label=domain_label)
            tar_batch_sampler_unlabeled = BatchSampler(tar_data_unlabeled, label_sampler, batch_size)
            tar_loader_unlabeled = torch.utils.data.DataLoader(tar_data_unlabeled, batch_sampler = tar_batch_sampler_unlabeled, num_workers=4)

            # 
            return loader_labeled_list, tar_loader_unlabeled
    else: 
        ## if phase=='test'
        src_loader_labeled_list = []
        domain_label = 0
        for labeled_source_list in labeled_source_list_array:
            src_data_test = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['test'], domain_label=domain_label)
            if is_extract == True:
                src_loader_labeled_list.append(torch.utils.data.DataLoader(src_data_test, batch_size=batch_size, shuffle=True, drop_last=False,
                                                 num_workers=4))
            else:
                src_loader_labeled_list.append(torch.utils.data.DataLoader(src_data_test, batch_size=batch_size, shuffle=False, drop_last=False,
                                                 num_workers=4))
            domain_label += 1

        tar_data_unlabeled = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['test'], type='labeled')
        tar_loader_unlabeled = torch.utils.data.DataLoader(tar_data_unlabeled, num_workers=4 , shuffle=False, drop_last=False)

        return src_loader_labeled_list, tar_loader_unlabeled