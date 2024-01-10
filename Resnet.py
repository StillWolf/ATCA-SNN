import torch.nn as nn
import math
from layer import *

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = tdLayer(nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False))
        self.bn1 = tdBatchNorm(nn.BatchNorm2d(planes), 1)
        self.conv2 = tdLayer(nn.Conv2d(planes, planes, 3, 1, 1, bias=False))
        self.bn2 = tdBatchNorm(nn.BatchNorm2d(planes), alpha=0.5 ** 0.5)

        self.stride = stride
        self.spike = LIF()

        self.downsample = tdBatchNorm(nn.BatchNorm2d(planes), alpha= 0.5 ** 0.5)

        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            self.downsample = nn.Sequential(
                tdLayer(nn.Conv2d(in_planes, planes * BasicBlock.expansion, 1, stride, bias=False)),
                tdBatchNorm(nn.BatchNorm2d(planes * BasicBlock.expansion), alpha=1/math.sqrt(2.))
            )

    def forward(self, x):
        out = self.spike(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.downsample(x) + out
        out = self.spike(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = tdLayer(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn1 = tdBatchNorm(nn.BatchNorm2d(planes))

        self.conv2 = tdLayer(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn2 = tdBatchNorm(nn.BatchNorm2d(planes))

        self.conv3 = tdLayer(nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False))
        self.bn3 = tdBatchNorm(nn.BatchNorm2d(planes * Bottleneck.expansion), alpha=1/math.sqrt(2.))

        self.downsample = tdBatchNorm(nn.BatchNorm2d(planes * Bottleneck.expansion), alpha=1 / math.sqrt(2.))

        self.spike = LIF()

        if stride != 1 or in_planes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                tdLayer(nn.Conv2d(in_planes, planes * Bottleneck.expansion, 1, stride, bias=False)),
                tdBatchNorm(nn.BatchNorm2d(planes * Bottleneck.expansion), alpha=1/math.sqrt(2.))
            )

    def forward(self, x):
        out = self.spike(self.bn1(self.conv1(x)))
        out = self.spike(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.downsample(x) + out
        out = self.spike(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channels = 2, if_dvs = False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.in_channels = in_channels
        self.if_dvs = if_dvs
        if if_dvs:
            self.conv1 = tdLayer(nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False))
            self.bn1 = TemporalBN(64, steps)  # tdBatchNorm(nn.BatchNorm2d(64))
        else:
            self.conv1 = (nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False))
            self.bn1 = (nn.BatchNorm2d(64))
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], False)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], True)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], True)
        #self.layer4 = self._make_layer(block, 512, num_block_layers[3], True)
        self.avg_pool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.voting1 = tdLayer(nn.Linear(512 * block.expansion, 256))
        self.voting2 = tdLayer(nn.Linear(256, num_classes))
        self.spike = LIF()

        # for m in self.modules():
        #     if isinstance(m, tdLayer):
        #         if isinstance(m.layer, nn.Conv2d):
        #             n = m.layer.kernel_size[0] * m.layer.kernel_size[1] * m.layer.out_channels
        #             m.layer.weight.data.normal_(0, math.sqrt(2. / n))
        #         elif isinstance(m.layer, nn.BatchNorm2d):
        #             m.layer.weight.data.fill_(1)
        #             m.layer.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, downsample):
        layers = []
        #down_sample = None
        if downsample:
            layers.append(block(self.inplanes, planes, 2))
        else:
            layers.append(block(self.inplanes, planes, 1))

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     down_sample = nn.Sequential(
        #         tdLayer(nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride = stride, bias=False)),
        #         tdBatchNorm(nn.BatchNorm2d(planes * block.expansion), alpha=1/math.sqrt(2.))
        #     )
        # else:
        #     down_sample = tdBatchNorm(nn.BatchNorm2d(planes), alpha=1 / math.sqrt(2.))

        #layers.append(block(self.inplanes, planes, stride, down_sample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.if_dvs:
            x = self.spike(self.bn1(self.conv1(x.float())))
            # x = self.spike(self.conv1(x.float()))
        else:
            x = self.bn1(self.conv1(x.float()))
            x, _ = torch.broadcast_tensors(x, torch.zeros((steps,) + x.shape))
            x = x.permute(1, 2, 3, 4, 0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avg_pool(out)

        out = out.view(out.shape[0], -1, out.shape[-1])

        out = self.spike(self.voting1(out))
        out, vmem_last = self.spike(self.voting2(out), True, True)
        # out, vmem = self.spike(out, True, True)
        out = self.spike(out, True, False)
        # return out, vmem, vmem_last
        return out


class Cifar_Net(nn.Module):  # Example net for CIFAR10
    def __init__(self, num_classes=11, in_channels=2, trace=False, size=32, basic_channel=64):
        super(Cifar_Net, self).__init__()
        self.basic_channel = basic_channel
        self.trace = trace
        self.bn0 = TemporalBN(self.basic_channel, steps)#tdBatchNorm(nn.BatchNorm2d(64))
        self.bn1 = TemporalBN(self.basic_channel*2, steps)#tdBatchNorm(nn.BatchNorm2d(128))
        self.bn2 = TemporalBN(self.basic_channel*4, steps)#tdBatchNorm(nn.BatchNorm2d(256))
        self.bn3 = TemporalBN(self.basic_channel*8, steps)#tdBatchNorm(nn.BatchNorm2d(512))
        self.bn4 = TemporalBN(self.basic_channel*4, steps)#tdBatchNorm(nn.BatchNorm2d(256))

        self.conv0_s = tdLayer(nn.Conv2d(in_channels, self.basic_channel, 3, 1, 1, bias=True))
        self.pool0_s = tdLayer(nn.AvgPool2d(2))
        self.conv1_s = tdLayer(nn.Conv2d(self.basic_channel, self.basic_channel*2, 3, 1, 1, bias=True))
        self.pool1_s = tdLayer(nn.AvgPool2d(2))
        self.conv2_s = tdLayer(nn.Conv2d(self.basic_channel*2, self.basic_channel*4, 3, 1, 1, bias=True))
        self.pool2_s = tdLayer(nn.AvgPool2d(2))
        self.conv3_s = tdLayer(nn.Conv2d(self.basic_channel*4, self.basic_channel*8, 3, 1, 1, bias=True))
        self.conv4_s = tdLayer(nn.Conv2d(self.basic_channel*8, self.basic_channel*4, 3, 1, 1, bias=True))

        self.fc1_s = tdLayer(nn.Linear((size//8) * (size//8) * self.basic_channel * 4, 512, bias=True))
        self.fc2_s = tdLayer(nn.Linear(512, num_classes, bias=True))

        self.spike = LIF()

    def forward(self, x):
        x = self.bn0(self.conv0_s(x.float()))
        x = self.spike(x)

        x = self.bn1(self.conv1_s(x))
        x = self.spike(x)

        x = self.pool1_s(x)

        x = self.bn2(self.conv2_s(x))
        x = self.spike(x)

        x = self.pool2_s(x)

        x = self.bn3(self.conv3_s(x))
        x = self.spike(x)

        x = self.bn4(self.conv4_s(x))
        x = self.spike(x)

        x = self.pool0_s(x)

        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x, vmem_last = self.spike(x, True, True)
        x = self.fc2_s(x)
        spike, vmem = self.spike(x, True, True)
        return spike, vmem, vmem_last


class Cifar_Net_NoBn(nn.Module):  # Example net for CIFAR10
    def __init__(self, num_classes=11, in_channels=2, trace=False, size=32, basic_channel=64, decay=True):
        super(Cifar_Net_NoBn, self).__init__()
        self.basic_channel = basic_channel
        self.trace = trace

        self.conv0_s = tdLayer(nn.Conv2d(in_channels, self.basic_channel, 3, 1, 1, bias=True))
        self.pool0_s = tdLayer(nn.AvgPool2d(2))
        self.conv1_s = tdLayer(nn.Conv2d(self.basic_channel, self.basic_channel * 2, 3, 1, 1, bias=True))
        self.pool1_s = tdLayer(nn.AvgPool2d(2))
        self.conv2_s = tdLayer(nn.Conv2d(self.basic_channel * 2, self.basic_channel * 4, 3, 1, 1, bias=True))
        self.pool2_s = tdLayer(nn.AvgPool2d(2))
        self.conv3_s = tdLayer(nn.Conv2d(self.basic_channel * 4, self.basic_channel * 8, 3, 1, 1, bias=True))
        self.conv4_s = tdLayer(nn.Conv2d(self.basic_channel * 8, self.basic_channel * 4, 3, 1, 1, bias=True))

        self.fc1_s = tdLayer(nn.Linear((size // 8) * (size // 8) * self.basic_channel * 4, 512, bias=True))
        self.fc2_s = tdLayer(nn.Linear(512, num_classes, bias=True))
        if decay:
            self.spike = C_LIF()
        else:
            self.spike = LIF()

    def forward(self, x):
        x = self.spike(self.conv0_s(x.float()))
        x = self.spike(self.conv1_s(x))
        x = self.pool1_s(x)
        x = self.spike(self.conv2_s(x))
        x = self.pool2_s(x)

        x = self.spike(self.conv3_s(x))

        x = self.spike(self.conv4_s(x))
        x = self.pool0_s(x)

        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x, vmem_last = self.spike(x, True, True)
        x = self.fc2_s(x)
        spike, vmem = self.spike(x, True, True)
        return spike, vmem, vmem_last


class Cifar_Net_Accumulate(nn.Module):  # Example net for CIFAR10
    def __init__(self, num_classes=11, in_channels=2, trace=False, size=32, basic_channel=64, decay=False):
        super(Cifar_Net_Accumulate, self).__init__()
        self.basic_channel = basic_channel
        self.trace = trace

        self.conv0_s = tdLayer(nn.Conv2d(in_channels, self.basic_channel, 3, 1, 1, bias=True))
        self.pool0_s = tdLayer(nn.AvgPool2d(2))
        self.conv1_s = tdLayer(nn.Conv2d(self.basic_channel, self.basic_channel * 2, 3, 1, 1, bias=True))
        self.pool1_s = tdLayer(nn.AvgPool2d(2))
        self.conv2_s = tdLayer(nn.Conv2d(self.basic_channel * 2, self.basic_channel * 4, 3, 1, 1, bias=True))
        self.pool2_s = tdLayer(nn.AvgPool2d(2))
        self.conv3_s = tdLayer(nn.Conv2d(self.basic_channel * 4, self.basic_channel * 8, 3, 1, 1, bias=True))
        self.conv4_s = tdLayer(nn.Conv2d(self.basic_channel * 8, self.basic_channel * 4, 3, 1, 1, bias=True))

        self.fc1_s = tdLayer(nn.Linear((size // 8) * (size // 8) * self.basic_channel * 4, 512, bias=True))
        self.fc2_s = tdLayer(nn.Linear(512, num_classes, bias=True))

        if decay:
            self.spike = C_LIF()
        else:
            self.spike = LIF()

    def forward(self, x):
        x = self.spike(self.conv0_s(x.float()))
        x = self.spike(self.conv1_s(x))
        x = self.pool1_s(x)
        x = self.spike(self.conv2_s(x))
        x = self.pool2_s(x)

        x = self.spike(self.conv3_s(x))

        x = self.spike(self.conv4_s(x))
        x = self.pool0_s(x)

        x = x.view(x.shape[0], -1, x.shape[4])
        x = self.fc1_s(x)
        x, vmem_last = self.spike(x, True, True)
        x = self.fc2_s(x)
        vmem = self.spike(x, True, False)
        spike = []
        return spike, vmem, vmem_last


class ResNet2(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=10, in_channels = 2, if_dvs = False):
        super(ResNet2, self).__init__()
        self.inplanes = 64
        self.in_channels = in_channels
        self.if_dvs = if_dvs
        if if_dvs:
            self.conv1 = tdLayer(nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False))
            self.bn1 = tdBatchNorm(nn.BatchNorm2d(64))
        else:
            self.conv1 = (nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False))
            self.bn1 = (nn.BatchNorm2d(64))
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], False)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], True)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], True)
        #self.layer4 = self._make_layer(block, 512, num_block_layers[3], True)
        self.avg_pool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.voting1 = tdLayer(nn.Linear(512 * block.expansion, 256))
        self.voting2 = tdLayer(nn.Linear(256, num_classes))
        self.spike = LIF()

        # for m in self.modules():
        #     if isinstance(m, tdLayer):
        #         if isinstance(m.layer, nn.Conv2d):
        #             n = m.layer.kernel_size[0] * m.layer.kernel_size[1] * m.layer.out_channels
        #             m.layer.weight.data.normal_(0, math.sqrt(2. / n))
        #         elif isinstance(m.layer, nn.BatchNorm2d):
        #             m.layer.weight.data.fill_(1)
        #             m.layer.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, downsample):
        layers = []
        #down_sample = None
        if downsample:
            layers.append(block(self.inplanes, planes, 2))
        else:
            layers.append(block(self.inplanes, planes, 1))

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     down_sample = nn.Sequential(
        #         tdLayer(nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride = stride, bias=False)),
        #         tdBatchNorm(nn.BatchNorm2d(planes * block.expansion), alpha=1/math.sqrt(2.))
        #     )
        # else:
        #     down_sample = tdBatchNorm(nn.BatchNorm2d(planes), alpha=1 / math.sqrt(2.))

        #layers.append(block(self.inplanes, planes, stride, down_sample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):

        if self.if_dvs:
            x = self.spike(self.bn1(self.conv1(x.float())))
        else:
            x = self.bn1(self.conv1(x.float()))
            x, _ = torch.broadcast_tensors(x, torch.zeros((steps,) + x.shape))
            x = x.permute(1, 2, 3, 4, 0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = self.avg_pool(out)

        out = out.view(out.shape[0], -1, out.shape[-1])

        out = self.spike(self.voting1(out))
        out, vmem_last = self.spike(self.voting2(out), True, True)
        out = self.spike(out, True, False)
        return out


class BasicBlock_NoBn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(BasicBlock_NoBn, self).__init__()
        self.conv1 = tdLayer(nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False))
        self.conv2 = tdLayer(nn.Conv2d(planes, planes, 3, 1, 1, bias=False))
        self.stride = stride
        self.spike = LIF()
        self.downsample = tdLayer(nn.Conv2d(in_planes, planes * BasicBlock.expansion, 1, stride, bias=False))

    def forward(self, x):
        out = self.spike(self.conv1(x))
        out = self.conv2(out)
        out = self.downsample(x) + out
        out = self.spike(out)
        return out


class ResNet_NoBn(nn.Module):
    def __init__(self, block, num_block_layers, num_classes=11, in_channels=2, if_dvs=False):
        super(ResNet_NoBn, self).__init__()
        self.inplanes = 64
        self.in_channels = in_channels
        self.if_dvs = if_dvs
        if if_dvs:
            self.conv1 = tdLayer(nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False))
            self.bn1 = TemporalBN(64, steps)  # tdBatchNorm(nn.BatchNorm2d(64))
        else:
            self.conv1 = (nn.Conv2d(self.in_channels, 64, 3, 1, 1, bias=False))
            self.bn1 = (nn.BatchNorm2d(64))
        self.layer1 = self._make_layer(block, 128, num_block_layers[0], False)
        self.layer2 = self._make_layer(block, 256, num_block_layers[1], True)
        self.layer3 = self._make_layer(block, 512, num_block_layers[2], True)
        # self.layer4 = self._make_layer(block, 512, num_block_layers[3], True)
        self.avg_pool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.voting1 = tdLayer(nn.Linear(512 * block.expansion, 256))
        self.voting2 = tdLayer(nn.Linear(256, num_classes))
        self.spike = LIF()

        # for m in self.modules():
        #     if isinstance(m, tdLayer):
        #         if isinstance(m.layer, nn.Conv2d):
        #             n = m.layer.kernel_size[0] * m.layer.kernel_size[1] * m.layer.out_channels
        #             m.layer.weight.data.normal_(0, math.sqrt(2. / n))
        #         elif isinstance(m.layer, nn.BatchNorm2d):
        #             m.layer.weight.data.fill_(1)
        #             m.layer.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, downsample):
        layers = []
        #down_sample = None
        if downsample:
            layers.append(block(self.inplanes, planes, 2))
        else:
            layers.append(block(self.inplanes, planes, 1))

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     down_sample = nn.Sequential(
        #         tdLayer(nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride = stride, bias=False)),
        #         tdBatchNorm(nn.BatchNorm2d(planes * block.expansion), alpha=1/math.sqrt(2.))
        #     )
        # else:
        #     down_sample = tdBatchNorm(nn.BatchNorm2d(planes), alpha=1 / math.sqrt(2.))

        #layers.append(block(self.inplanes, planes, stride, down_sample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.spike(self.conv1(x.float()))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.avg_pool(out)

        out = out.view(out.shape[0], -1, out.shape[-1])

        out = self.spike(self.voting1(out))
        out, vmem_last = self.spike(self.voting2(out), True, True)
        out, vmem = self.spike(out, True, True)
        # out = self.spike(out, True, False)
        return out, vmem, vmem_last
        # return out


def resnet19():
    return ResNet(BasicBlock, [3, 3, 2], 11, if_dvs=True)


def resnet19_NoBn():
    return ResNet_NoBn(BasicBlock_NoBn, [3, 3, 2], 11, if_dvs=True)

# def resnet34():
#     return ResNet(BasicBlock, [3, 4, 5, 3], 10)
#
# def resnet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3], 10)


if __name__ == '__main__':
    x = torch.ones((1, 2, 32, 32, 200), dtype=torch.float32, device=device)
    snn = Cifar_Net_NoBn(decay=False)
    snn = snn.to(device)
    spike, vmem, vmem_last = snn(x)
    print(vmem.shape)
