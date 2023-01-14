import torch.nn as nn
import torch
from yolo import YoloLayer


class ConvBR(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3,
                 stride=1, paddding=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c_in, c_out,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            paddding=paddding, bias=False),
                                  nn.BatchNorm2d(c_out),
                                  nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)


class DWConv3x3(nn.Module):
    def __init__(self, c_in, c_out, stride,
                 padding, use_bn=True, activation='relu'):
        super(DWConv3x3, self).__init__()

        self.d_conv = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride,
                                padding=padding, groups=c_in, bias=False)
        self.w_conv = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn = nn.Identity()
        if use_bn:
            self.bn = nn.BatchNorm2d(c_out)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(1e-2, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.d_conv(x)
        x = self.w_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, stride=1, padding=1,
                 use_bn=True, activation='relu'):
        super(ResBlock, self).__init__()
        self.conv0 = ConvBR(c_in=c_in, c_out=c_mid,
                            kernel_size=1, stride=1, paddding=0)
        self.conv1 = DWConv3x3(c_in=c_mid, c_out=c_out, stride=stride,
                               padding=padding, use_bn=use_bn, activation=activation)
        self.short_cut = nn.Identity()
        if stride != 1:
            self.short_cut = ConvBR(c_in, c_out, 1, stride, 0)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x)
        shot_cut = self.short_cut(x)
        out = x2 + shot_cut
        return out


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, c_mid, stride, repeat=3):
        super(ConvBlock, self).__init__()
        self.conv0 = ConvBR(c_in, c_mid, 1, 1, 0)
        self.block = self.make_layers(c_mid, stride, repeat)
        self.conv1 = ConvBR(c_mid, c_out, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.block(x)
        x = self.conv1(x)
        return x

    def make_layers(self, c_mid, stride, repeat):
        block = [ ]
        for i in range(repeat):
            if i == 0:
                block.append(ResBlock(c_mid, c_mid // 2, c_mid,
                                      stride=stride, padding=1))
            else:
                block.append(ResBlock(c_mid, c_mid // 2, c_mid, stride=1, padding=1))
        return nn.Sequential(*block)


class UpSuampeBlock(nn.Module):
    def __init__(self, c_in, c_out, up_scale=2):
        super(UpSuampeBlock, self).__init__()
        self.conv = nn.Sequential(nn.Upsample(scale_factor=up_scale,
                                              mode='bilinear', align_corners=True),
                                  ConvBR(c_in, c_out, 1, 1, 0))
    def forward(self, x):
        return self.conv(x)


class DarkNet(nn.Module):
    def __init__(self, num_classes):
        super(DarkNet, self).__init__()
        self.feature_len = num_classes + 5
        self.conv0 = ConvBR(c_in=3, c_out=64, kernel_size=7, stride=1, paddding=3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, padding=2)
        self.conv1 = ResBlock(c_in=64, c_mid=32, c_out=64, stride=2, padding=1)
        self.block0 = ConvBlock(c_in=64, c_out=128, c_mid=64, stride=2, repeat=2)
        self.block = ConvBlock(c_in=128, c_out=128, c_mid=256, stride=2, repeat=4)
        self.up_sample = UpSuampeBlock(c_in=128, c_out=128, up_scale=2)
        self.cat_conv = DWConv3x3(c_in=256, c_out=128, stride=1, padding=1)
        self.final_conv = ConvBR(c_in=128, c_out=self.feature_len, kernel_size=1,
                                 stride=1, paddding=0)
        self.yolo_layer = YoloLayer()

    def forward(self, x, y):
        x = self.conv0(x)
        x = self.max_pool(x)
        scale_feature1 = self.block0(x)
        scale_feature2 = self.block(scale_feature1)
        scale_feature3 = self.block(scale_feature2)
        up_feature2 = self.up_sample(scale_feature3)
        feature2 = torch.cat([up_feature2, scale_feature2], dim=1)
        feature2 = self.cat_conv(feature2)
        feature1 = torch.cat([feature2, scale_feature1], dim=1)
        feature1 = self.cat_conv(feature1)

        yolo_feature1 = self.final_conv(feature1)
        yolo_feature2 = self.final_conv(feature2)
        yolo_feature3 = self.final_conv(scale_feature3)
        output1 = self.yolo_layer(yolo_feature1, y)
        output2 = self.yolo_layer(yolo_feature2, y)
        output3 = self.yolo_layer(yolo_feature3, y)
        outputs = torch.cat([output1[0], output2[0], output3[0]], dim=1)
        if y is not None:
            losses = output1[1] + output2[1] + output3[1]
            return (outputs, losses)
        else:
            return outputs,

