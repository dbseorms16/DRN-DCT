# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import torch
import torch.nn as nn
from model import common, dct
import numpy as np
import torch.nn.functional as nnf
import math

def make_model(opt, parent=False):
    return RDN(opt)

class H2A2SR(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(H2A2SR, self).__init__()
        self.scale = opt.scale[0]
        self.int_scale = math.floor(self.scale)
        self.float_scale = opt.float_scale
        self.total_scale = opt.total_scale
        self.res_scale = self.total_scale / self.int_scale
        kernel_size = 3
        act = nn.ReLU(True)

        self.dct = dct.DCT_2D()
        self.idct = dct.IDCT_2D()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size, padding=1)
        self.R1 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R2 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R3 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R4 = common.RCAB(conv, 64, kernel_size, act=act)
        self.R5 = common.RCAB(conv, 64, kernel_size, act=act)
        self.t = nn.Conv2d(64, 3, kernel_size, padding=1)

        self.cof = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.cof_relu = nn.ReLU()
        
        self.x_cof = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.x_cof_relu = nn.ReLU()

    def forward(self, x, outH, outW):
        _,_, h, w = x.size()
        # th, tw = int(h / self.res_scale) , int(w / self.res_scale)
        # x = nnf.interpolate(x, size=(th, tw), mode='bicubic', align_corners=False).to('cuda:0')
        
        x = self.dct(x)
        zeroPad2d = nn.ZeroPad2d((0, outW-w, 0, outH-h)).to('cuda:0')

        # zeroPad2d = nn.ZeroPad2d((0, w-tw, 0, h-th)).to('cuda:0')

        x = zeroPad2d(x)

        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        # mask = torch.ones((th, tw), dtype=torch.int64, device = torch.device('cuda:0'))
        #diagonal 이 양수일 경우 대각선 라인이 왼쪽 상단으로 향함
        diagonal = w-2
        ## lf, hf 나누기
        lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        
        lf_mask = zeroPad2d(lf_mask)
        hf_mask = zeroPad2d(hf_mask)
        
        lf_mask = lf_mask.unsqueeze(0).expand(x.size())
        hf_mask = hf_mask.unsqueeze(0).expand(x.size())

        dlf = x * lf_mask
        dhf = x * hf_mask
        # dhf = dhf * coefficent
        # plf = pd.DataFrame(dlf[0,0,:,:].to('cpu').numpy())
        # phf = pd.DataFrame(dhf[0,0,:,:].to('cpu').numpy())
        # plf.to_csv('lf.csv', index=False)
        # phf.to_csv('hf.csv', index=False)

        ## 모델 시작
        ## 고주파 집중네트워크
        x = self.idct(x)

        hf = self.idct(dhf)
        coefficent = self.x_cof_relu(self.x_cof)

        x_cof = self.cof_relu(self.cof)
        x = x * x_cof
        hf = hf * coefficent
        hf = self.conv1(hf)
        hf = self.R1(hf)
        hf = self.R2(hf)
        hf = self.R3(hf)
        hf = self.R4(hf)
        hf = self.R5(hf)
        hf = self.t(hf)

        ## 다음에는 마스크빼보자
        #no dct 
        #그냥 x를 학습 했을때 35.17
        #나눠서 x를 학습 했을때 27.16
        ## 계수 합치기

        # result = x + hf
        result = x + hf
        return result

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, opt):
        super(RDN, self).__init__()
        #r은 스케일
        r = 2
        G0 = 64
        kSize = 3

        self.scale = opt.scale[0]
        self.int_scale = math.floor(self.scale)
        self.float_scale = opt.float_scale
        self.total_scale = opt.total_scale
        self.res_scale = self.total_scale / self.int_scale
        self.h2a2sr = H2A2SR(opt)

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = 16, 8, 64


        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(opt.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, opt.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, opt.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x, outH, outW):
        _,_, h, w = x.size()
        th, tw = int(h / self.res_scale) , int(w / self.res_scale)
        x = nnf.interpolate(x, size=(th, tw), mode='bicubic', align_corners=False).to('cuda:0')
        ##여기서 바이큐빅으로 줄이기
        #원본 32.06
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        x = self.UPNet(x)
        x = self.add_mean(x)
        x = self.h2a2sr(x, outH, outW)

        return x
