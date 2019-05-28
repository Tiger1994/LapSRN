import torch
import torch.nn as nn
import numpy as np
import math


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.relu = nn.ReLU()

    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)


class RDB(nn.Module):
    def __init__(self,G0,C,G,kernel_size = 3):
        super(RDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G,G))
        self.conv = nn.Sequential(*convs)
        #local_feature_fusion
        self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size = 1,padding = 0,stride =1)

    def forward(self,x):
        out = self.conv(x)
        lff = self.LFF(out)
        #local residual learning
        return lff + x


class rdn(nn.Module):
    def __init__(self, opts):
        '''
        opts: the system para
        '''
        super(rdn,self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0:local and global feature fusion layers 64filter
        '''
        self.D = opts.D
        self.C = opts.C
        self.G = opts.G
        self.G0 = opts.G0
        print ("D:{},C:{},G:{},G0:{}".format(self.D,self.C,self.G,self.G0))
        kernel_size =opts.kernel_size
        input_channels = opts.input_channels
        #shallow feature extraction
        self.SFE1 = nn.Conv2d(input_channels,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride=  1)
        self.SFE2 = nn.Conv2d(self.G0,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride =1)
        #RDB for paper we have D RDB block
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
        #Global feature fusion
        self.GFF = nn.Sequential(
               nn.Conv2d(self.D*self.G0,self.G0,kernel_size = 1,padding = 0 ,stride= 1),
               nn.Conv2d(self.G0,self.G0,kernel_size,padding = kernel_size>>1,stride = 1),
        )
        #upsample net
        self.up_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.G, out_channels=self.G, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        #init
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.kaiming_normal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()

    def forward(self, x):
        out  = self.SFE2(x)
        RDB_outs = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs, 1)
        out = self.GFF(out)
        out = x+out
        return self.up_net(out)


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=opt.G, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R1 = nn.Conv2d(in_channels=opt.G, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(rdn(opt))

        self.convt_I2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R2 = nn.Conv2d(in_channels=opt.G, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(rdn(opt))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.conv_input(x))

        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1

        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2

        return HR_2x, HR_4x


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss