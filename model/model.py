import torch
import torch.nn as nn

from model.basenet import network_dict
from loss.loss import  H_Distance
from utils import globalvar as gl

class RRL(nn.Module):

    def __init__(self, basenet, n_class, bottleneck_dim, domain_labels = None):
        super(RRL, self).__init__()
        self.basenet = network_dict[basenet]()
        self.basenet_type = basenet
        self._in_features = self.basenet.len_feature()
        
        if self.basenet_type.lower() not in ['resnet18_']:
            self.bottleneck = nn.Sequential(
                nn.Linear(self._in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                # nn.InstanceNorm1d(bottleneck_dim), 
                nn.ReLU(inplace=True)
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_dim, n_class)
        else:
            self.fc = nn.Linear(self._in_features, n_class)

        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()
        self.domain_labels = domain_labels

    def forward(self, source, target=None, source_label=None, target_label=None, source_domain_label = None, target_domain_label=None):
        DEVICE = gl.get_value('DEVICE')
        source_features = self.basenet(source)
        if self.basenet_type.lower() not in ['resnet18_']:
            source_features = self.bottleneck(source_features)
        source_output = self.fc(source_features)
       
        loss = 0

        if self.training == True and target is None and source_domain_label is not None:
            pass

        elif target is not None and target_domain_label is not None:
            # calculate the probability
            softmax_layer = nn.Softmax(dim=1).to(DEVICE)
            target_features = self.basenet(target)
            if self.basenet_type.lower() not in ['resnet18_']:
                target_features = self.bottleneck(target_features)
            target_output = self.fc(target_features)
            target_softmax = softmax_layer(target_output)
            target_prob, target_l = torch.max(target_softmax, 1)
            features = torch.cat([source_features,target_features])
            labels = torch.cat([source_label,target_l]) 
            domain_label = torch.cat([source_domain_label,target_domain_label])
            loss = H_Distance(features, labels, domain_label, device=DEVICE)
        return source_output, loss
    
    def get_bottleneck_features(self, inputs):
        features = self.basenet(inputs)
        return self.bottleneck(features)

    def get_fc_features(self, inputs):
        features = self.basenet(inputs)
        features = self.bottleneck(features)
        return self.fc(features)


