import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision

class AlexNetF(nn.Module):
    def __init__(self):
        super(AlexNetF, self).__init__()
        model_alexnet = torchvision.models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.avgpool = model_alexnet.avgpool
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(str(i), model_alexnet.classifier[i])
        self._in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        for i in range(6):
            x = self.classifier[i](x)
        return x
    
    def len_feature(self):
        return self._in_features
    
class VGGF(nn.Module):
    def __init__(self):
        super(VGGF, self).__init__()    
        model_vgg16 = torchvision.models.vgg16(pretrained=True)
        self.features = model_vgg16.features
        self.avgpool = model_vgg16.avgpool
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(str(i), model_vgg16.classifier[i])
        self._in_features = model_vgg16.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        for i in range(6):
            x = self.classifier[i](x)
        return x
    
    def len_feature(self):
        return self._in_features

class ResNet18F(nn.Module):
    
    def __init__(self):
        super(ResNet18F, self).__init__()    
        # model_resnet18 = torchvision.models.resnet18(pretrained=False)
        # model_resnet18.load_state_dict(torch.load('../resnet18-5c106cde.pth'))
        model_resnet18 = torchvision.models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self._in_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def len_feature(self):
        return self._in_features

class ResNet34F(nn.Module):
    
    def __init__(self):
        super(ResNet34F, self).__init__()    
        model_resnet34 = torchvision.models.resnet34(pretrained=True)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self._in_features = model_resnet34.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def len_feature(self):
        return self._in_features
    
class ResNetF(nn.Module):
    
    def __init__(self):
        super(ResNetF, self).__init__()    
        model_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self._in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def len_feature(self):
        return self._in_features


class ResNet101F(nn.Module):
    
    def __init__(self):
        super(ResNet101F, self).__init__()    
        model_resnet101 = torchvision.models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self._in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def len_feature(self):
        return self._in_features

network_dict = {"alexnet": AlexNetF,
                "vgg16": VGGF,
                "resnet18": ResNet18F,
                "resnet34": ResNet34F,
                "resnet": ResNetF,
                "resnet101": ResNet101F}



class Classifier(nn.Module):

    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.fc(x)
        return out

class ReverseLayerF(Function):
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):

    def __init__(self, input_dims, hidden_dims=3072, output_dims=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dims, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, output_dims),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class BaseNet(nn.Module):
    
    def __init__(self, basenet, n_class):
        super(BaseNet, self).__init__()
        self.basenet = network_dict[basenet]()
        self._in_features = self.basenet.len_feature()
        self.fc = nn.Linear(self._in_features, n_class)

        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        features = self.basenet(x)
        source_output = self.fc(features)

        return source_output, None

    def get_features(self, x):
        features = self.basenet(x)

        return features
