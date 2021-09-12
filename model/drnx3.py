import torch
import torch.nn as nn
from model import common
import numpy as np
import torch.nn.functional as nnf


def make_model(opt):
    return DRN(opt)


class New1(nn.Module):
    def __init__(self, num_channels=3):
        super(New1, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
# class SRCNN(nn.Module):
#     def __init__(self, num_channels=3):
#         super(SRCNN, self).__init__()
#         self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
#         self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
#         return x

class new2(nn.Module):
    def __init__(self, num_channels=3):
        super(new2, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv1x1_1 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv1x1_2 = nn.Conv2d(32, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.conv1x1_3 = nn.Conv2d(num_channels, num_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.tail = nn.Conv2d(num_channels * 2 , num_channels, kernel_size=3, padding= 3 // 2)

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.cat((x, res), 1)
        x = self.tail(x)
        return x

class NewCNN(nn.Module):
    def __init__(self, opt, num_channels=3, conv=common.default_conv):
        super(NewCNN, self).__init__()
        self.opt = opt
        self.n_feats = opt.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.RCAB1 = common.RCAB(conv, self.n_feats * pow(2, 1), kernel_size, act=act)
        self.RCAB2 = common.RCAB(conv, 64, kernel_size, act=act)
        self.RCAB3 = common.RCAB(conv, 128, kernel_size, act=act)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(256, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.tail = nn.Conv2d(num_channels , num_channels, kernel_size=3, padding= 3 // 2)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        copyx = x
        x = self.RCAB1(x)
        x = torch.cat((x, copyx), 1)
        x = self.RCAB2(x)
        copyx2 = x
        x = torch.cat((x, copyx2), 1)
        x = self.RCAB3(x)
        copyx3 = x
        x = torch.cat((x, copyx3), 1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tail(x)
        return x

class DCT(nn.Module):
    def __init__(self, opt):
        super(DCT, self).__init__()
        self.int_scale = max(opt.scale)
        self.float_scale = opt.float_scale
        self.scale = self.int_scale + self.float_scale
        self.res_scale = self.scale / self.int_scale
        

    def dct_2d(self, x, norm=None):
        X1 = self.dct(x, norm=norm)
        X2 = self.dct(X1.transpose(-1, -2), norm=norm)
        return X2.transpose(-1, -2)

    def idct_2d(self, X, norm=None):
        x1 = self.idct(X, norm=norm)
        x2 = self.idct(x1.transpose(-1, -2), norm=norm)
        return x2.transpose(-1, -2)

    def _rfft(self, x, signal_ndim=1, onesided=True):
        # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
        # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
        # torch 1.8.0 torch.fft.rfft to torch 1.5.0 torch.rfft as signal_ndim=1
        # written by mzero
        odd_shape1 = (x.shape[1] % 2 != 0)
        x = torch.fft.rfft(x)
        x = torch.cat([x.real.unsqueeze(dim=2), x.imag.unsqueeze(dim=2)], dim=2)
        if onesided == False:
            _x = x[:, 1:, :].flip(dims=[1]).clone() if odd_shape1 else x[:, 1:-1, :].flip(dims=[1]).clone()
            _x[:,:,1] = -1 * _x[:,:,1]
            x = torch.cat([x, _x], dim=1)
        return x

    def _irfft(self, x, signal_ndim=1, onesided=True):
        # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
        # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
        # torch 1.8.0 torch.fft.irfft to torch 1.5.0 torch.irfft as signal_ndim=1
        # written by mzero
        if onesided == False:
            res_shape1 = x.shape[1]
            x = x[:,:(x.shape[1] // 2 + 1),:]
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x, n=res_shape1)
        else:
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x)
        return x

    def dct(self, x, norm=None):
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = self._rfft(v, 1, onesided=False)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def idct(self, X, norm=None):
        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = self._irfft(V, 1, onesided=False)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)

    def forward(self, x):
        N, C, H, W = x.size()
        outH, outW = int(H*self.res_scale), int(W*self.res_scale)
        x = self.dct_2d(x)
        # x = x[:, :, 0:int(H*self.scale), 0:int(W*self.scale)]
        zeroPad2d = nn.ZeroPad2d((0,int(outW - W), 0, int(outH - H))).to('cuda:0')
        x = zeroPad2d(x)
        x = self.idct_2d(x)
        return x

class DRN(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(DRN, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3
        self.dct = DCT(opt)
        self.NewCNN = NewCNN(opt)
        
        sf= 0
        if (self.scale[0]%2) ==0:
            sf =2
        elif self.scale[0] ==3:
            sf =3
        print('DRN scale >> '+str(sf))

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt, self.scale[0], n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
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

    def forward(self, x):

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
            copy = copies[self.phase - idx - 1]

            # concat down features and upsample features

            if self.opt.scale[0] ==3:
                x =nnf.interpolate(x, size=(len(copy[0][0]),len(copy[0][0][0])), mode='bicubic', align_corners=False)
            
            x = torch.cat((x, copies[self.phase - idx - 1]), 1)

            # output sr imgs
            sr = self.tail[idx + 1](x)
            sr = self.add_mean(sr)
            sr = self.dct(sr)
            sr = self.NewCNN(sr)
            results.append(sr)

        return results