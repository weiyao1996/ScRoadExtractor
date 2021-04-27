"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class ResNet34_EdgeNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet34_EdgeNet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.aspp = _ASPP(512, [1, 2, 4])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.b_conv1 = nn.Conv2d(512, 128, 3, padding=1)
        self.b_conv2 = nn.Conv2d(256, 64, 3, padding=1)
        self.b_conv3 = nn.Conv2d(64, 1, 1)
        self.b_conv4 = nn.Conv2d(128, 64, 1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x) # 128*128*64
        e2 = self.encoder2(e1) # 64*64*128
        e3 = self.encoder3(e2) # 32*32*256
        e4 = self.encoder4(e3) # 16*16*512

        # Center
        e4 = self.aspp(e4) # 16*16*512

        # Decoder
        d4 = self.decoder4(e4) + e3 # 32*32*256
        d3 = self.decoder3(d4) + e2 # 64*64*128
        d2 = self.decoder2(d3) + e1 # 128*128*64
        d1 = self.decoder1(d2) # 256*256*64

        # EdgeNet Block
        up1 = F.interpolate(e4, size=[64, 64], mode="bilinear") # 64*64*512
        up1 = self.b_conv1(up1) # 64*64*128
        up_concat = torch.cat([up1, e2], dim=1) # 64*64*256
        up2 = F.interpolate(up_concat, size=[256, 256], mode="bilinear") # 256*256*256
        up2 = self.b_conv2(up2) # 256*256*64
        edge_map = F.interpolate(up2, size=[512, 512], mode="bilinear") # 512*512*64
        edge = self.b_conv3(edge_map) # 512*512*1

        atten_concat = torch.cat([d1, up2], dim=1) # 256*256*128
        result = self.b_conv4(atten_concat)# 256*256*64
        out = self.finaldeconv1(result)
        out = self.finalrelu1(out)
        out = self.finalconv2(out) # 512*512*32
        out = self.finalrelu2(out)
        out = self.finalconv3(out) # 512*512*1

        return torch.sigmoid(out), edge


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(_ASPP, self).__init__()
        out_channels = in_channels
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3)
        self.b4 = _AsppPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


pretrained_mean = torch.tensor([0.485, 0.456, 0.406], requires_grad = False).view((1, 3, 1, 1))
pretrained_std = torch.tensor([0.229, 0.224, 0.225], requires_grad = False).view((1, 3, 1, 1))


if __name__=="__main__":
    input=torch.ones(1,3,512,512)
    net=ResNet34_EdgeNet()
    output=net.forward(input)
    print(output.size())
