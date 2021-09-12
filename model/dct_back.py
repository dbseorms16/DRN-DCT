import torch
import torch.nn as nn
from model import common, dct
import numpy as np
import torch.nn.functional as nnf
import math
import pandas as pd

def make_model(opt):
    return DRN(opt)
class H2A2SR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(H2A2SR, self).__init__()
        self.scale = args.scale[0]
        self.int_scale = math.floor(self.scale)
        self.float_scale = args.float_scale
        self.total_scale = args.total_scale
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

    def forward(self, x, outH, outW):
        _,_,h,w = x.size()
        
        x = self.dct(x)
        zeroPad2d = nn.ZeroPad2d((0, outW-w, 0, outH-h)).to('cuda:0')

        x = zeroPad2d(x)

        mask = torch.ones((outH, outW), dtype=torch.int64, device = torch.device('cuda:0'))
        #diagonal 이 양수일 경우 대각선 라인이 왼쪽 상단으로 향함
        
        diagonal = (outW-w-2) if (outW-w) > (outH-h) else (outH-h-2)
        ## lf, hf 나누기

        lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1
        lf_mask = lf_mask.unsqueeze(0).expand(x.size())
        lf = x * lf_mask

        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(x.size())
        hf = x * hf_mask
        
        # dlf = pd.DataFrame(lf[0,0,:,:].to('cpu').numpy())
        # dlf.to_csv('lf.csv', index=False)

        # df = pd.DataFrame(hf[0,0,:,:].to('cpu').numpy())
        # df.to_csv('hf.csv', index=False)
        ## 모델 시작
        ## 고주파 집중네트워크
        lf = self.idct(lf)
        hf = self.idct(hf)

        hf = self.conv1(hf)
        hf = self.R1(hf)
        hf = self.R2(hf)
        hf = self.R3(hf)
        hf = self.R4(hf)
        hf = self.R5(hf)
        hf = self.t(hf)

        result = lf + hf
        return result

class DRN(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(DRN, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3
        self.h2a2sr = H2A2SR(opt)
        
        self.scale = opt.scale[0]
        self.int_scale = math.floor(self.scale)
        self.float_scale = opt.float_scale
        self.total_scale = opt.total_scale
        self.res_scale = self.total_scale / self.int_scale
        sf= 0
        if (self.int_scale%2) ==0:
            sf =2
        elif self.int_scale ==3:
            sf =3
        print('DRN scale >> '+str(sf))

        self.upsample = nn.Upsample(scale_factor=opt.int_scale,
                                    mode='bicubic', align_corners=False)
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt, self.int_scale, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        up_body_blocks = [[
            common.RCAB(
                conv, n_feats * pow(2, p), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks.insert(0, [
            common.RCAB(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])

        # The first upsample block
        up = [[
            common.Upsampler(conv, sf, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, sf, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats * pow(2, p), opt.n_colors, kernel_size)
            )
        self.tail = nn.ModuleList(tail)

        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x, outH, outW):
        _,_, h, w = x.size()
        th, tw = int(h / self.res_scale) , int(w / self.res_scale)
        x = nnf.interpolate(x, size=(th, tw), mode='bicubic', align_corners=False).to('cuda:0')

        # upsample x to target sr size
        x = self.upsample(x)
        # preprocess
        x = self.sub_mean(x)

        x = self.head(x)
        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)


        # up phases
        sr = self.tail[0](x)
        sr = self.add_mean(sr)
        results = [sr]
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)

            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)
            sr = self.h2a2sr(sr, outH, outW)

            results.append(sr)

        return results