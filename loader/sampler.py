import random
import torch

class LabelSampler(object):
    def __init__(self, batch_num, class_num, select_class_num):
        self.class_num = class_num
        self.select_class_num = select_class_num
        self.set_batch_num(batch_num)

    def set_batch_num(self, batch_num):
        self.labels = []
        while len(self.labels) < batch_num:
            labels = random.sample(list(range(self.class_num)), self.select_class_num)
            self.labels.append(labels)
        print('LabelSampler batch_num:{}'.format(len(self.labels)))

    def __getitem__(self, index):
        return self.labels[index]
    
    def __iter__(self):
        return iter(self.labels)
    
    def __len__(self):
        return len(self.labels)

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, label_sampler, batch_size):
        self.dataset = dataset
        self.label_sampler = label_sampler 
        if batch_size % label_sampler.select_class_num != 0:
            raise ValueError('batch_size error')
        self.batch_size = batch_size
        self.per_num = batch_size // label_sampler.select_class_num
    def __iter__(self):
        indexes = []
        for labels in self.label_sampler:
            idxs = self.dataset.get_indexes_by_labels(labels, self.per_num)
            indexes.append(idxs)

        for idx_batch in indexes:
            yield idx_batch

    def __len__(self):
        return len(self.label_sampler)