import numpy as np
import os
import random
import torch
from PIL import Image

class ImageList(torch.utils.data.Dataset):
    def __init__(self, file_name, root_dir, transform=None, type=None, domain_label=None):
        super(ImageList, self).__init__()
        self.transform = transform
        self.type = type

        self.images, self.labels = [], []
        self.class_labels = []
        self.pseudo_labels = []
        self.domain_label = domain_label
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for item in lines:
                line = item.strip().split(' ')
                self.images.append(os.path.join(root_dir, line[0]))
                self.labels.append(int(line[1].strip()))
                # if domain_label is not None:
                    # self.domain_labels.append(domain_label)
            
        self.class_num = len(set(self.labels))
        self.class_labels = [ [] for i in range(self.class_num)]
        for idx, label in enumerate(self.labels):
            self.class_labels[label].append(idx)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        if self.domain_label is not None:
            return img, target, self.domain_label
        else:
            return img, target

    def __len__(self):
        return len(self.images)
    
    def get_indexes_by_labels(self, labels, per_num): 
        indexes = []
        if self.type == 'labeled':
            for label in labels:
                idx = random.sample(self.class_labels[label], per_num if len(self.class_labels[label]) > per_num else len(self.class_labels[label]))
                indexes += idx
        elif self.type == 'unlabeled':
            if len(self.pseudo_labels) == 0:
                raise ValueError('pseudo_labels must be computed before calling "get_indexes_by_labels(...)" when "task" is target!')
            for label in labels:
                idx = random.sample(self.pseudo_labels[label], per_num if len(self.pseudo_labels[label]) > per_num else len(self.pseudo_labels[label]))
                indexes += idx

        return indexes
        
    def update_pseudo_labels(self, labels):
        if len(self.labels) != len(labels):
            raise ValueError('Error! len of pseudo labels is not equal to len of labels!')
        self.pseudo_labels = [ [] for i in range(self.class_num)]
        for idx, label in enumerate(labels):
            self.pseudo_labels[label].append(idx)       
        correct, total = 0, 0
        for pseudo_label, true_label in zip(labels, self.labels):
            total += 1
            if pseudo_label == true_label:
                correct += 1
                
        len_of_class = []
        for i in range(self.class_num):
            len_of_class.append((i,len(self.pseudo_labels[i])))
        print(len_of_class)
        print('Update pseudo labels. Accuracy is {:.4f}, {}/{}.'.format(correct / total, correct, total))

        
class ImageCLEF(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super(ImageCLEF, self).__init__()
        self.transform = transform
        file_name = os.path.join(root_dir,'list',domain + 'List.txt')
        lines = open(file_name, 'r').readlines()
        self.images, self.labels = [], []
        self.domain = domain
        for item in lines:
            line = item.strip().split(' ')
            self.images.append(os.path.join(root_dir, domain, line[0].split('/')[-1]))
            self.labels.append(int(line[1].strip()))

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target  #  return img, target, index

    def __len__(self):
        return len(self.images)



class MutilImageList(torch.utils.data.Dataset):
    def __init__(self, files_name, root_dir, transform=None, type=None):
        super(MutilImageList, self).__init__()
        self.transform = transform
        self.type = type

        self.images, self.labels = [], []
        self.class_labels = []
        self.pseudo_labels = []
        if isinstance(files_name, str):
            files_name = [files_name]
        for file_name in files_name:
            with open(file_name, 'r') as f:
                lines = f.readlines()
                for item in lines:
                    line = item.strip().split(' ')
                    self.images.append(os.path.join(root_dir, line[0]))
                    label = int(line[1].strip())
                    if 'kfold' in file_name: ## pacs
                        label -= 1
                    self.labels.append(label)
        data = list(zip(self.images, self.labels))
        random.shuffle(data)
        self.images, self.labels = zip(*data)

        self.class_num = len(set(self.labels))
        self.class_labels = [ [] for i in range(self.class_num)]
        for idx, label in enumerate(self.labels):
            self.class_labels[label].append(idx)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)
    