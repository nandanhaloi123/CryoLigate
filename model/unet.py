# Code adapted from https://github.com/AghdamAmir/3D-UNet/blob/main/unet3d.py

from torch import nn
# from torchsummary import summary
import torch
import time
import math

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, bottleneck = False) -> None:
        super(Conv3DBlock, self).__init__()
        
        self.time_mlp =  nn.Linear(time_emb_dim, out_channels//2)
        
        self.conv1 = nn.Conv3d(in_channels= in_channels, out_channels=out_channels//2, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
        self.conv2 = nn.Conv3d(in_channels= out_channels//2, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)


    def forward(self, input, t):
        res = self.relu(self.bn1(self.conv1(input)))
        
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 3]
        # print(res.size(), time_emb.size())
        res = res + time_emb
        
        res = self.relu(self.bn2(self.conv2(res)))
        
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res




class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, time_emb_dim, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        
        self.time_mlp =  nn.Linear(time_emb_dim, in_channels)
        
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, t, residual=None):
        out = self.upconv1(input)
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 3]
        out = out + time_emb
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512, time_emb_dim = 32) -> None:
        super(UNet3D, self).__init__()
        
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        
        
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
                
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls, time_emb_dim=time_emb_dim)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls, time_emb_dim=time_emb_dim)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls, time_emb_dim=time_emb_dim)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, time_emb_dim=time_emb_dim, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, time_emb_dim=time_emb_dim, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, time_emb_dim=time_emb_dim, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, time_emb_dim=time_emb_dim, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    
    def forward(self, input, timestep):
        #Analysis path forward feed
        t = self.time_mlp(timestep)
        print(t.size())
        out, residual_level1 = self.a_block1(input, t)
        out, residual_level2 = self.a_block2(out, t)
        out, residual_level3 = self.a_block3(out, t)
        out, _ = self.bottleNeck(out, t)

        #Synthesis path forward feed
        out = self.s_block3(out, t, residual_level3)
        out = self.s_block2(out, t, residual_level2)
        out = self.s_block1(out, t, residual_level1)
        return out





# level_channels=[64, 128, 256]
# bottleneck_channel=512
# in_channels = 1
# level_1_chnls = level_channels[0]
# level_2_chnls = level_channels[1]
# level_3_chnls = level_channels[2]
# a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
# a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
# a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
# bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)

# out, residual_level1 = a_block1(data)
# print(out.size(), residual_level1.size())
# out, residual_level2 = a_block2(out)
# print(out.size(), residual_level2.size())
# out, residual_level3 = a_block3(out)
# print(out.size(), residual_level3.size())
# out, _ = bottleNeck(out)
# print("Bottleneck", out.size())
# num_classes = 1
# s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
# s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
# s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

# out = s_block3(out, residual_level3)
# print(out.size())
# out = s_block2(out, residual_level2)
# print(out.size())
# out = s_block1(out, residual_level1)
# print(out.size())
