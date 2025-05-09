import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
# from utils.lossFunctions import *


class DWConv(nn.Module):
    def __init__(self, dim=768,group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim//group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.gn1=nn.GroupNorm(hidden_features//4,hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.gn2 = nn.GroupNorm(hidden_features // 4, hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.gn3=nn.GroupNorm(out_features//4,out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x=self.gn1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x=self.gn2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x=self.gn3(x)
        x = self.drop(x)
        return x


class LocalSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=7):
        super(LocalSABlock, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.GroupNorm(k*heads//4,k*heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
            nn.GroupNorm(k*u//4,k*u)
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.GroupNorm(self.vv*u//4,self.vv*u)
        )

        self.softmax = nn.Softmax(dim=-1)

        self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h)  # b, heads, k , w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h))  # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h)  # b, v, uu, w * h
        content = torch.einsum('bkum,bvum->bkv', (softmax, values))
        content = torch.einsum('bhkn,bkv->bhvn', (queries, content))
        values = values.view(n_batch, self.uu, -1, w, h)
        context = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
        context = context.view(n_batch, self.kk, self.vv, w * h)
        context = torch.einsum('bhkn,bkvn->bhvn', (queries, context))

        out = content + context
        out = out.contiguous().view(n_batch, -1, w, h)

        return out


class TFBlock(nn.Module):

    def __init__(self, in_chnnels, out_chnnels, mlp_ratio=2., drop=0.3,
                 drop_path=0., act_layer=nn.GELU, linear=False):
        super(TFBlock, self).__init__()
        self.in_chnnels = in_chnnels
        self.out_chnnels = out_chnnels
        self.attn = LocalSABlock(
            in_channels=in_chnnels, out_channels=out_chnnels
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        hidden_planes = max(planes,in_planes) // self.expansion
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(hidden_planes //4,
                                hidden_planes)  
        self.conv2 = nn.ModuleList([TFBlock(hidden_planes, hidden_planes)])
        self.bn2 = nn.GroupNorm(hidden_planes // 4,
                                hidden_planes)  
        self.conv2.append(nn.GELU()) 
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(planes // 4, planes)  
        self.GELU=nn.GELU()
        self.shortcut = nn.Sequential()
        if in_planes!=planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.GroupNorm(planes//4,planes)
            )
    def forward(self, x):
        out = self.GELU(self.bn1(self.conv1(x)))  
        out = self.conv2(out)
        out = self.GELU(self.bn3(self.conv3(out))) 
        out += self.shortcut(x)
        return out


class Trans_EB(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Bottleneck(in_, out)
        self.activation=torch.nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class LABlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LABlock, self).__init__()
        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(output_channels//4,output_channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.gelu=nn.GELU()
    def forward(self, inputs):
        sum = 0
        for input in inputs:
            sum += input
        sum=self.gelu(sum)
        out = self.W_1(sum)
        psi = self.psi(out)  # Mask
        return psi


class Fuse(nn.Module):

    def __init__(self, nn, hidden_dims, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(hidden_dims, 1)

    def forward(self, down_inp, up_inp, attention):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = self.nn(outputs)
        outputs = attention * outputs
        outputs = self.conv(outputs)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        return outputs

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, layer_num, conv_layer=False):
        super(Down, self).__init__()
        # self.conv = ConvRelu(in_channels, out_channels)
        transformer_list = []

        for i in range(layer_num):
            if i==0:
                if conv_layer:
                    transformer_list.append(ConvRelu(in_channels, out_channels))
                else:
                    transformer_list.append(Trans_EB(in_channels, out_channels))
            else:
                transformer_list.append(Trans_EB(out_channels, out_channels))
        self.layers = nn.ModuleList(transformer_list)
        self.patch_embed = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x_trans_list = []
        for layer in self.layers:
            x = layer(x)
            x_trans_list.append(x)

        outputs, indices = self.patch_embed(x)
        return outputs, indices, x_trans_list

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num, conv_layer=False):
        super().__init__()
        # self.
        transformer_list = []

        for i in range(layer_num):
            if i==layer_num-1:
                if conv_layer:
                    transformer_list.append(ConvRelu(in_channels, out_channels))
                else:
                    transformer_list.append(Trans_EB(in_channels, out_channels))
            else:
                transformer_list.append(Trans_EB(in_channels, in_channels))
        self.layers = nn.ModuleList(transformer_list)

        self.inv_patch_embed = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, x, indices):
        x = self.inv_patch_embed(x, indices=indices)

        x_trans_list = []
        for layer in self.layers:
            x = layer(x)
            x_trans_list.append(x)

        return x, x_trans_list[:-1]

class crackformer(nn.Module):

    def __init__(self, in_channels, final_hidden_dims=64, num_classes=2):
        super(crackformer, self).__init__()

        self.in_channels = in_channels
        self.stage = 5
        self.layer_num = [2, 2, 3, 3, 3]
        self.embed_dims = [64, 128, 256, 512, 512]
        self.hidden_dims = final_hidden_dims
        self.num_classes = num_classes
        down_list = []

        self.down1 = Down(in_channels=3, out_channels=self.embed_dims[0], layer_num=self.layer_num[0], conv_layer=True)
        self.down2 = Down(in_channels=self.embed_dims[0], out_channels=self.embed_dims[1], layer_num=self.layer_num[1], conv_layer=False)
        self.down3 = Down(in_channels=self.embed_dims[1], out_channels=self.embed_dims[2], layer_num=self.layer_num[2], conv_layer=False)
        self.down4 = Down(in_channels=self.embed_dims[2], out_channels=self.embed_dims[3], layer_num=self.layer_num[3], conv_layer=False)
        self.down5 = Down(in_channels=self.embed_dims[3], out_channels=self.embed_dims[4], layer_num=self.layer_num[4], conv_layer=False)

        self.up5 = Up(in_channels=self.embed_dims[4], out_channels=self.embed_dims[3], layer_num=self.layer_num[4], conv_layer=False)
        self.up4 = Up(in_channels=self.embed_dims[3], out_channels=self.embed_dims[2], layer_num=self.layer_num[3], conv_layer=False)
        self.up3 = Up(in_channels=self.embed_dims[2], out_channels=self.embed_dims[1], layer_num=self.layer_num[2], conv_layer=False)
        self.up2 = Up(in_channels=self.embed_dims[1], out_channels=self.embed_dims[0], layer_num=self.layer_num[1], conv_layer=False)
        self.up1 = Up(in_channels=self.embed_dims[0], out_channels=self.embed_dims[0], layer_num=self.layer_num[0], conv_layer=False)


        self.fuse5 = Fuse(ConvRelu(self.embed_dims[4] + self.embed_dims[3], self.hidden_dims), self.hidden_dims, scale=16)
        self.fuse4 = Fuse(ConvRelu(self.embed_dims[3] + self.embed_dims[2], self.hidden_dims), self.hidden_dims, scale=8)
        self.fuse3 = Fuse(ConvRelu(self.embed_dims[2] + self.embed_dims[1], self.hidden_dims), self.hidden_dims, scale=4)
        self.fuse2 = Fuse(ConvRelu(self.embed_dims[1] + self.embed_dims[0], self.hidden_dims), self.hidden_dims, scale=2)
        self.fuse1 = Fuse(ConvRelu(self.embed_dims[0] + self.embed_dims[0], self.hidden_dims), self.hidden_dims, scale=1)

        self.final = Conv1X1(5, num_classes)

        self.LABlock_1 = LABlock(self.embed_dims[0], self.hidden_dims)
        self.LABlock_2 = LABlock(self.embed_dims[1], self.hidden_dims)
        self.LABlock_3 = LABlock(self.embed_dims[2], self.hidden_dims)
        self.LABlock_4 = LABlock(self.embed_dims[3], self.hidden_dims)
        self.LABlock_5 = LABlock(self.embed_dims[4], self.hidden_dims)

    # def calculate_loss(self, outputs, labels):
    #     loss = 0
    #     loss = cross_entropy_loss_RCF(outputs, labels)
    #     return loss

    def forward(self, inputs):

        # encoder part
        out1, indices_1, en_list1 = self.down1(inputs)
        out2, indices_2, en_list2 = self.down2(out1)
        out3, indices_3, en_list3 = self.down3(out2)
        out4, indices_4, en_list4 = self.down4(out3)
        out5, indices_5, en_list5 = self.down5(out4)

        # decoder part
        x_de5, de_list5 = self.up5(out5, indices=indices_5)
        x_de4, de_list4 = self.up4(x_de5, indices=indices_4)
        x_de3, de_list3 = self.up3(x_de4, indices=indices_3)
        x_de2, de_list2 = self.up2(x_de3, indices=indices_2)
        x_de1, de_list1 = self.up1(x_de2, indices=indices_1)

        # attention part
        attention1 = self.LABlock_1(en_list1[:-1] + de_list1)
        attention2 = self.LABlock_2(en_list2[:-1] + de_list2)
        attention3 = self.LABlock_3(en_list3[:-1] + de_list3)
        attention4 = self.LABlock_4(en_list4[:-1] + de_list4)
        attention5 = self.LABlock_5(en_list5[:-1] + de_list5)

        # fuse part
        fuse5 = self.fuse5(down_inp=en_list5[-1], up_inp=x_de5, attention=attention5)
        fuse4 = self.fuse4(down_inp=en_list4[-1], up_inp=x_de4, attention=attention4)
        fuse3 = self.fuse3(down_inp=en_list3[-1], up_inp=x_de3, attention=attention3)
        fuse2 = self.fuse2(down_inp=en_list2[-1], up_inp=x_de2, attention=attention2)
        fuse1 = self.fuse1(down_inp=en_list1[-1], up_inp=x_de1, attention=attention1)

        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))

        return [fuse5, fuse4, fuse3, fuse2, fuse1], output


if __name__ == '__main__':
    inp = torch.randn(10, 3, 360, 640)
    model = crackformer(3, 64)
    _, out=model(inp)
    print(model)
    print(out.shape)

