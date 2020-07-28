import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from _collections import OrderedDict
from models.model_uitls import init_weights


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove `module`
        new_state_dict[name] = v
    return new_state_dict


class Classifier(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class, dropout=0.5):
        super(Classifier, self).__init__()
        # solution 1
        self.layer1 = nn.Dropout(0.5)
        self.layer2 = nn.Sequential(nn.Linear(in_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU())

        self.layer3 = nn.Linear(mid_channel, n_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.layer2(x)
        pred = self.layer3(x)
        return pred


class ResNet_SIIM(nn.Module):
    def __init__(self, in_channel=3, n_class=1):
        super(ResNet_SIIM, self).__init__()

        resnet = models.resnet34(True)
        self.layer0 = nn.Sequential(resnet.conv1,
                                    resnet.bn1,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc_reduce = nn.Linear(resnet.fc.in_features, 32)
        # self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = Classifier(32, 32, n_class)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # self.layer0.eval()
        self.layer1.eval()
        self.layer2.eval()
        self.layer3.eval()
        self.layer4.eval()

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc_reduce(x.squeeze())
        x = self.classifier(x)

        return x


if __name__ == '__main__':

    input_seq = torch.rand(10, 3, 224, 224)
    resnet = ResNet_SIIM()
    model_state_dict = torch.load('/media/zyi/080c2d74-aa6d-4851-acdc-c9588854e17c/medical_AI/HAM_pre_trained/cnn-hm10000_best.model',
                                  map_location='cpu')['model_state_dict']
    for k, v in model_state_dict.items():
        if 'classifier' in k:
            print(k)

    resnet.load_state_dict(convert_state_dict(model_state_dict))

    output = resnet(input_seq)
    print(output)
