import torch
import torch.nn as nn
from models.regnet_ import regnet_800M
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
    def __init__(self, in_channel, mid_channel, n_class):
        super(Classifier, self).__init__()
        # solution 1
        self.layer1 = nn.Dropout(0.3)
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


class Regnet_SIIM(nn.Module):
    def __init__(self, in_channel=3, n_class=1):
        super(Regnet_SIIM, self).__init__()
        backbone = regnet_800M(pretrained=True)
        self.stem = list(list(backbone.children())[0].children())[0]
        self.feature = nn.Sequential(*list(list(backbone.children())[0].children())[1:-1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(672, 320, n_class)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # self.layer0.eval()
        self.feature.eval()    # freeze bn
        x = self.stem(x)
        x = self.feature(x)

        x = self.avgpool(x)
        x = self.classifier(x.squeeze())

        return x


if __name__ == '__main__':

    input_seq = torch.rand(10, 3, 256, 256)
    resnet = Regnet_SIIM()
    print(resnet)
    output = resnet(input_seq)
    print(output.shape)
    # efficient = EfficientNet.from_pretrained('efficientnet-b2')
    # a = efficient.extract_features(input_seq)
    # print('ss ', a.shape)