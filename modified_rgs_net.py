%%writefile d_rgsnet.py
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from group_norm  import GroupNorm2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def norm2d(planes, num_channels_per_group=32):
    print("num_channels_per_group:{}".format(num_channels_per_group))
    if num_channels_per_group > 0:
        return GroupNorm2d(planes, num_channels_per_group, affine=True,
                           track_running_stats=False)
    else:
        return nn.BatchNorm2d(planes)

class skew_normal(nn.Module):
    def __init__(self, channels, height):
        super(skew_normal,self).__init__()
        self.channels = channels
                                                                                                              
        self.height = height
        if self.height>=64:
            self.projection = nn.Conv2d(channels,channels, kernel_size = 3, stride=2, padding=1)
            self.upsample = nn.ConvTranspose2d(channels, channels, kernel_size = 3, stride=2, padding=1)
            self.height_1 = int(height/2)

        else:
            self.projection = None
            self.upsample = None
            self.height_1 = height

        self.channel_delta=nn.Parameter(torch.rand(self.channels,self.height_1,self.height_1))
        self.tanh = nn.Tanh()
    def forward(self, x):
        if self.projection is not None:
            out = self.projection(x)
        else:
            out = x
        #batch=[]
        '''for i in range(out.size()[0]):
            proxy = torch.randn(1)
            transform_channel=[]
            for j in range(self.channels):
                transform_channel.append(out[i][j]*torch.sqrt(1-self.channel_delta[j]*self.channel_delta[j])+self.channel_delta[j]*proxy)

            batch.append(torch.stack(transform_channel))'''
        '''max_element = torch.max(self.channel_delta).item()
        min_element = torch.max(self.channel_delta).item()
        if min_element < 0:
            min_element = -min_element
        if min_element > max_element:
            scale = min_element+1
        else:
            scale = max_element+1
        #shift_ = self.channel_delta*2-1
        shift_ = self.channel_delta/scale'''
        shift_ = self.tanh(self.channel_delta)
        channel_weight = torch.cat(out.size()[0]*[shift_.unsqueeze(0)])
        out = torch.mul(out,channel_weight)
        if self.upsample is not None:
            out = self.upsample(out, output_size = x.size())
        '''else:
            new = torch.stack(batch)'''
        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 group_norm=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(planes, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.relu1=nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes, group_norm)
        self.downsample = downsample
        self.stride = stride
        #self.conv3=nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)#edit
        self.bn3 = norm2d(planes,group_norm)
    def forward(self, x):
        residual = x
        print(x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(x.size())
        out = self.conv2(out)
        out = self.bn2(out)

        '''
        if self.downsample is not None:
            residual = self.downsample(x)'''
        if self.downsample is not None:
            residual = self.downsample(residual)
        '''if residual.shape != out.shape:
            residual=self.conv3(residual)'''
        rgs=self.relu1(residual)
        rgs=self.bn3(rgs)
        #out += residual
        
        out += residual
        out = self.relu(out)

        return out


'''class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 group_norm=0):
        super(Bottleneck, self).__init__()
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        #self.bn1 = norm2d(planes, group_norm)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        #self.bn2 = norm2d(planes, group_norm)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        #self.bn3 = norm2d(planes * 4, group_norm)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        
        #self.bn4= norm2d(planes * 4, group_norm)
        self.bn4 = nn.GroupNorm(32 , planes * 4)
        #self.bn5= norm2d(planes,group_norm)
        self.bn5 = nn.GroupNorm(32 , planes)
        #self.bn6 = nn.BatchNorm2d(planes)
        #self.bn7 = nn.BatchNorm2d(planes * 4)
        self.gelu = nn.GELU()
    def forward(self, x):
        residual = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)

        
        
        
        #edit
        
        d_rgs=self.gelu(out)
        #d_rgs=self.relu(d_rgs)
        d_rgs=self.bn5(d_rgs)
        #d_rgs=self.bn6(d_rgs)
        #d_rgs_gelu = self.gelu(out)

        #edit
        
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += d_rgs
        

        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        residual=self.gelu(residual)
        #rgs=self.relu(residual)
        rgs=self.bn4(residual)
        #rgs=self.bn7(rgs)
        #rgs_gelu = self.gelu(residual)
        #out += residual
#        if self.downsample is not None:
#           residual = self.downsample(rgs)
        out += rgs
        #out = self.relu(out)

        return out
        
'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dim, binary, stride=1, downsample=None,
                 group_norm=0):
        super(Bottleneck, self).__init__()
        '''self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm2d(planes * 4, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride'''
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.bn4 = nn.GroupNorm(32 , planes *4 )
        self.bn5 = nn.GroupNorm(32 , planes)
        #self.bn6 = nn.BatchNorm2d(planes)
        #self.bn7 = nn.BatchNorm2d(planes * 4)
        self.height = dim
        self.transform = skew_normal(planes * 4 ,dim)
#        self.transform_1 = skew_normal(planes ,dim)
    def forward(self,x):
        residual = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)

        #d_rgs_skew = self.transform_1(out)
        #residual_2 = out
        d_rgs = self.relu(out)
        d_rgs = self.bn5(d_rgs)
        #my_rgs =d_rgs_skew * .6 + d_rgs *4


        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + d_rgs

        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        #residual_2 = residual
        residual_1 = self.transform(residual)
        residual = self.relu(residual)
        residual = self.bn4(residual)
        out = out + residual * .4 + residual_1 * .6 
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, group_norm=0):#block = basic_block or bottleneck_block,layers=a list containing the block number descriptions
        self.inplanes = 64
        super(ResNet, self).__init__()
        #for cifar
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        #for mnist and Imagenet
        '''self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)'''
        #self.bn1 = norm2d(64, group_norm)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 32,0, layers[0],
                                       group_norm=group_norm)
        self.layer2 = self._make_layer(block, 128, 16,1, layers[1], stride=2,
                                       group_norm=group_norm)
        self.layer3 = self._make_layer(block, 256, 8,0, layers[2], stride=2,
                                       group_norm=group_norm)
        self.layer4 = self._make_layer(block, 512, 4,1, layers[3], stride=2,
                                       group_norm=group_norm)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GroupNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, dim, binary, blocks, stride=1, group_norm=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=3, stride=stride,padding = 1, bias=False),
                #norm2d(planes * block.expansion, group_norm),
                #nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,dim,binary, stride, downsample,
                            group_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,dim,binary, group_norm=group_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet26(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasciBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
