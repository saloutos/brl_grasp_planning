import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# resnets without MinkowskiEngine

### RESNET BASE ###

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self,
                inplanes,
                planes,
                stride=1,
                dilation=1,
                downsample=None,
                bn_momentum=0.1,
                dimension=3):
        super(BasicBlock3D, self).__init__()
        assert dimension > 0

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation)
        self.norm1 = nn.BatchNorm3d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, dilation=dilation)
        self.norm2 = nn.BatchNorm3d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.inplanes, kernel_size=3, stride=2),
            nn.InstanceNorm3d(self.inplanes), # TODO: 3D? or 2D or 1D?
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2
        )
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2
        )
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2
        )
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2
        )

        self.conv5 = nn.Sequential(
            nn.Dropout(), # TODO: change p=0.5 and inplace?
            nn.Conv3d(self.inplanes, self.inplanes, kernel_size=3, stride=3),
            nn.InstanceNorm3d(self.inplanes), # TODO: again, is 3D correct?
            nn.GELU(),
        )

        # TODO: not sure how to handle this, need to get shape of layer before and after?
        # NOTE: might not need to worry about this, since we are modifying for ResUNet later
        self.glob_pool = nn.MaxPool3d(self.inplanes)

        self.final = nn.Linear(self.inplanes, out_channels, bias=True)

    # NOTE: removed weight initialization!

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.glob_pool(x)
        return self.final(x)


### BACKBONE RESUNET14 ###

class ResUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=5)

        self.bn0 = nn.BatchNorm3d(self.inplanes)

        self.conv1p1s2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                        self.LAYERS[0])

        self.conv2p2s2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm3d(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                        self.LAYERS[1])

        self.conv3p4s2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2)

        self.bn3 = nn.BatchNorm3d(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                        self.LAYERS[2])

        self.conv4p8s2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=2, stride=2)
        self.bn4 = nn.BatchNorm3d(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                        self.LAYERS[3])

        self.convtr4p16s2 = nn.ConvTranspose3d(self.inplanes, self.PLANES[4], kernel_size=2, stride=2)
        self.bntr4 = nn.BatchNorm3d(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                        self.LAYERS[4])
        self.convtr5p8s2 = nn.ConvTranspose3d(self.inplanes, self.PLANES[5], kernel_size=2, stride=2)
        self.bntr5 = nn.BatchNorm3d(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                        self.LAYERS[5])
        self.convtr6p4s2 = nn.ConvTranspose3d(self.inplanes, self.PLANES[6], kernel_size=2, stride=2)
        self.bntr6 = nn.BatchNorm3d(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                        self.LAYERS[6])
        self.convtr7p2s2 = nn.ConvTranspose3d(self.inplanes, self.PLANES[7], kernel_size=2, stride=2)
        self.bntr7 = nn.BatchNorm3d(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                        self.LAYERS[7])

        self.final = nn.Conv3d(self.PLANES[7] * self.BLOCK.expansion, out_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        out = torch.cat((out, out_b3p8), dim=0) # TODO: check dims!!
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = torch.cat((out, out_b2p4), dim=0) # TODO: check dims!!
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = torch.cat((out, out_b1p2), dim=0) # TODO: check dims!!
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = torch.cat((out, out_p1), dim=0) # TODO: check dims!!
        out = self.block8(out)

        return self.final(out)

class ResUNet14(ResUNetBase):
    BLOCK = BasicBlock3D
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)

class ResUNet14D(ResUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 192, 192)