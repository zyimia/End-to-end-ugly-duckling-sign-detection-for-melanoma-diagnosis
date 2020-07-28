import os
import torch
import math
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict
import torchvision.models as models
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import load_state_dict_from_url, model_urls


def get_classifier_para():
    model_para = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-kaggle_MIC/kaggle_skin/cnn-kaggle_best.model')['model_state_dict']
    encoder_para_keys = []
    classifier_para_keys = []
    for k, v in model_para.items():
        if 'combiner' not in k:
            encoder_para_keys.append(k)
        else:
            classifier_para_keys.append(k)

    for k in encoder_para_keys:
        del model_para[k]

    new_state_dict = OrderedDict()
    for k in classifier_para_keys:
        new_state_dict[k[16:]] = model_para[k]

    return new_state_dict


class Classifier(nn.Module):
    def __init__(self, in_channel, mid_channel, n_class):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU(inplace=False))

        self.layer2 = nn.Sequential(nn.Dropout(0.6),
                                    nn.Linear(mid_channel, mid_channel),
                                    nn.BatchNorm1d(mid_channel),
                                    nn.ReLU(inplace=False))

        self.layer3 = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(mid_channel, n_class))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm1d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # x = nn.functional.dropout(x, p=0.4)
        x = self.layer1(x)
        x = self.layer2(x)
        pred = self.layer3(x)

        return pred


def cov_feature(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h*w
    x = x.reshape(batchsize, dim, M)
    I_hat = (-1./M/M)*torch.ones(dim, dim, device=x.device) + (1./M)*torch.eye(dim, dim, device=x.device)
    I_hat = I_hat.view(1, dim, dim).repeat(batchsize, 1, 1).type(x.dtype)
    y = (x.transpose(1, 2)).bmm(I_hat).bmm(x)
    return y


def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    """
    https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks_other.py
    :param net:
    :param init_type:
    :return:
    """
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class TimeDistributed(nn.Module):
    def __init__(self, modules, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = modules
        self.batch_first = batch_first
        for para in self.module.parameters():
            para.requires_grad = False

    def forward(self, x_seq):
        assert len(x_seq.size()) >= 4
        if len(x_seq.size()) == 4:
            x_seq = x_seq.unsqueeze(0)

        batch, time_step, C, H, W = x_seq.shape

        features = []
        for t in range(time_step):
            x = self.module(x_seq[:, t, ...])
            features.append(x.view(batch, -1))  # N*C

        features = torch.stack(features, dim=1)   # N*T*C

        return features


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, add_pool=False):
        super(convbnrelu, self).__init__()
        self.add_pool = add_pool
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        if self.add_pool:
            self.pool = nn.MaxPool2d(kernel_size=kernel, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.add_pool:
            x = self.pool(x)
        return x


class SE_block(nn.Module):
    def __init__(self, in_channel, ratio=8, block_type='scSE'):
        """
        :param in_channel:
        :param ratio:
        :param block_type: 'scSE', 'sSE', 'cSE'
        https://arxiv.org/pdf/1808.08127.pdf
        https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation.py
        """
        super(SE_block, self).__init__()
        assert block_type in ('scSE', 'sSE', 'cSE')

        self.type = block_type
        self.channel_se = False if self.type == 'sSE' else True
        self.spatial_se = False if self.type == 'cSE' else True

        # spatial squeeze & channel excitation
        if self.channel_se:
            out_channel = in_channel//ratio
            self.cse_fc0 = nn.Linear(in_channel, out_channel)
            self.relu = nn.ReLU(inplace=True)
            self.cse_fc1 = nn.Linear(out_channel, in_channel)
            self.sigmoid = nn.Sigmoid()

        # channel squeeze
        if self.spatial_se:
            self.sse_conv = nn.Conv2d(in_channel, 1, kernel_size=1, stride=1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        batch_size, num_channels, H, W = inputs.size()
        output = inputs

        if self.channel_se:
            output_cse = inputs.view(batch_size, num_channels, -1).mean(dim=2)   # global average
            output_cse = self.relu(self.cse_fc0(output_cse))     # fc --> relu
            output_cse = self.sigmoid(self.cse_fc1(output_cse))  # fc --> sigmoid
            # channel-wise multiple
            output_cse = torch.mul(inputs, output_cse.view(batch_size, num_channels, 1, 1))
            output = output_cse

        if self.spatial_se:
            output_sse = self.sigmoid(self.sse_conv(inputs))     # conv --> sigmoid

            # spatially multiple
            output_sse = torch.mul(inputs, output_sse.view(batch_size, 1, H, W))
            output = output_sse

        if self.type == 'scSE':
            # otuput = output_cse + output_sse
            output = torch.max(output_cse, output_sse)

        return output


class SKinnet(nn.Module):
    def __init__(self):
        super(SKinnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 2, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.PReLU()
        self.max_pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                                    nn.BatchNorm2d(32),
                                    nn.PReLU())

        self.layer2 = nn.Sequential(nn.MaxPool2d(3, 2, 1),
                                    nn.Conv2d(32, 32, 3, 1, 1),
                                    nn.BatchNorm2d(32),
                                    nn.PReLU())

        self.layer3 = nn.Sequential(nn.MaxPool2d(3, 2, 1),
                                    nn.Conv2d(32, 32, 3, 1, 1),
                                    nn.BatchNorm2d(32),
                                    nn.PReLU())

        self.layer4 = nn.Sequential(nn.MaxPool2d(3, 2, 1),
                                    nn.Conv2d(32, 32, 3, 1, 1),
                                    nn.BatchNorm2d(32),
                                    nn.PReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, diff=None):
        if diff is not None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x += diff[0]

            x = self.max_pool(x)

            x = self.layer1(x)
            x += diff[1]

            x = self.layer2(x)
            x += diff[2]

            x = self.layer3(x)
            x += diff[3]

            x = self.layer4(x)
            # x += diff[4]
            x = self.avg_pool(x)

            return x
        else:
            features = []

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            features.append(x)

            x = self.max_pool(x)

            x = self.layer1(x)
            features.append(x)

            x = self.layer2(x)
            features.append(x)

            x = self.layer3(x)
            features.append(x)

            x = self.layer4(x)
            # features.append(x)
            # x = self.avg_pool(x)

            return x, features


class ResNet(models.ResNet):
    def __init__(self, block, layers):
        super(ResNet, self).__init__(block=block, layers=layers)
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def _forward(self, x, diff=None):
        if diff is not None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x += diff[0]

            x = self.maxpool(x)

            x = self.layer1(x)
            x += diff[1]

            x = self.layer2(x)
            x += diff[2]

            x = self.layer3(x)
            x += diff[3]

            x = self.layer4(x)
            # x += diff[4]
            # x = F.dropout2d(x, p=0.5)
            # x = self.avgpool(x)

            return x
        else:
            features = []

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            features.append(x)

            x = self.maxpool(x)

            x = self.layer1(x)
            features.append(x)

            x = self.layer2(x)
            features.append(x)

            x = self.layer3(x)
            features.append(x)

            x = self.layer4(x)
            # features.append(x)
            # x = F.dropout2d(x, p=0.5)
            # x = self.avgpool(x)

            return x, features

    forward = _forward


def load_pretrained_resnet(arch, progress=True):
    if arch == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    if arch == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2])

    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    model.load_state_dict(state_dict, strict=False)
    model.fc_channel = model.fc.in_features
    delattr(model, 'fc')

    return model


def load_finetuned_resnet(arch, layer_reduce=False):
    if arch == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3])
    if arch == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2])

    if layer_reduce:
        state_dict = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-hm10000_MIC/dropout123456_reduce2'
                                '/cnn-hm10000_best.model')['model_state_dict']
    else:
        state_dict = torch.load('/home/zyi/MedicalAI/Skin_lesion_prognosis/run_exp/cnn-hm10000_MIC/dropout123456/'
                                'cnn-hm10000_best.model')['model_state_dict']

    conv1 = OrderedDict()
    conv1['weight'] = state_dict['module.layer0.0.weight']

    bn1 = OrderedDict()
    bn1['weight'] = state_dict['module.layer0.1.weight']
    bn1['bias'] = state_dict['module.layer0.1.bias']
    bn1['running_mean'] = state_dict['module.layer0.1.running_mean']
    bn1['running_var'] = state_dict['module.layer0.1.running_var']

    layer1 = OrderedDict()
    for k, v in state_dict.items():
        if 'layer1' in k:
            layer1[k[14:]] = v

    layer2 = OrderedDict()
    for k, v in state_dict.items():
        if 'layer2' in k:
            layer2[k[14:]] = v

    layer3 = OrderedDict()
    for k, v in state_dict.items():
        if 'layer3' in k:
            layer3[k[14:]] = v

    layer4 = OrderedDict()
    for k, v in state_dict.items():
        if 'layer4' in k:
            layer4[k[14:]] = v

    model.conv1.load_state_dict(conv1)
    model.bn1.load_state_dict(bn1)
    model.layer1.load_state_dict(layer1)
    model.layer2.load_state_dict(layer2)
    model.layer3.load_state_dict(layer3)
    model.layer4.load_state_dict(layer4)

    if layer_reduce:
        reduce_layer_dict = OrderedDict()
        reduce_layer_dict['weight'] = state_dict['module.fc_reduce.weight']
        reduce_layer_dict['bias'] = state_dict['module.fc_reduce.bias']
        return model, reduce_layer_dict
    else:
        return model


if __name__ == '__main__':
    # inputs = torch.rand(2, 3, 120, 120)
    # model = GSOP_block(3)
    # out = model(inputs)

    model = load_pretrained_resnet('resnet34')
    print(list(model.layer4.parameters())[-1])
