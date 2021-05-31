import torch
import numpy as np
import pandas as pd
import cv2
import utility
from decimal import Decimal
from tqdm import tqdm
from option import args
import os
from torchvision import transforms 
from torchvision.utils import save_image 
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import math
import imageio
def dct_2d( x, norm=None):

    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d( X, norm=None):

    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)
    

def _rfft( x, signal_ndim=1, onesided=True):
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

def _irfft( x, signal_ndim=1, onesided=True):
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

def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = _rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def idct(X, norm=None):
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

    v = _irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)
class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.float_scale = opt.float_scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8

    ######by given the scale and the size of input image
    ######we caculate the input matrix for the weight prediction network
    ###### input matrix for weight prediction network
    

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        int_scale = max(self.scale)
        float_scale = self.float_scale
        scale = int_scale + float_scale
        res_scale = scale / int_scale 
        self.ckp.set_epoch(epoch)


        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        for name, param in self.model.named_parameters():
            splitname = name.split('.')
            if splitname[1] != 'P2W':
                param.requires_grad = False


        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, idx) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
           

            N,C,H,W = lr[0].size()
            outH,outW = int(H*scale),int(W*scale)

            scale_coord_map, mask = self.input_matrix_wpn(H * int_scale,W * int_scale, res_scale)
            scale_coord_map = scale_coord_map.to("cuda:0")

            sr = self.model(lr[0], scale_coord_map)

            srDct = dct_2d(sr[-1])
            srDct = srDct[:, :, 0:int(outH), 0:int(outW)]
            srDct = srDct * ((scale*scale)/16)
            re_sr  = idct_2d(srDct)

            # re_sr = torch.nn.functional.interpolate(sr[-1], (outH, outW),
            #                  mode='bilinear',align_corners=True )
            
            # re_sr= re_sr.contiguous().view(N, C,outH,outW)
            # re_sr = utility.quantize(re_sr, self.opt.rgb_range).type(torch.int)
            sr[-1] = re_sr


            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[(i-1) - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss
            ##여기서 sr사이즈 hr사이즈
            loss_primary = self.loss(sr[-1], hr)
            loss_primary += self.loss(sr[0], lr[0])
            
            # compute dual loss
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            # compute average loss
            # average_feat =(sr[-1]+fflip_sr[-1])/2
            # loss_average = self.loss(average_feat, hr)

            

            #copute flip loss
            loss_flip =0
            # for i in range(0, len(sr)):
            #     loss_flip+= self.loss(sr[i], fflip_sr[i])

            # compute total loss
            loss =  loss_primary+ self.opt.dual_weight * loss_dual
            # loss =  loss_primary+ self.opt.dual_weight 
            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()
    def input_matrix_wpn(self,inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        outH, outW = int(scale*inH), int(scale*inW)
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH,  scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)
        if add_scale:
            scale_mat = torch.zeros(1,1)
            scale_mat[0,0] = 1.0/scale
            #res_scale = scale_int - scale
            #scale_mat[0,scale_int-1]=1-res_scale
            #scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  ###(inH*inW*scale_int**2, 4)
            
        ####projection  coordinate  and caculate the offset 
        h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag,  0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1
        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)
        # print('pos_mat')
        # print(pos_mat)
        # print(inW,inH)
        return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW
    
    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))

        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            int_scale = max(self.scale)
            float_scale = self.float_scale
            scale = int_scale + float_scale
            res_scale = scale / int_scale
            for si, s in enumerate([int_scale]):
                eval_psnr = 0
                eval_simm =0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)


                    # lr[0] = torch.rand(1,3,2,2)
                    N,C,H,W = lr[0].size()
                    # print(lr[0].size())

                    outH,outW = int(H*scale),int(W*scale)
                    #_,_,outH,outW = hr.size()
                    #timer_test.tic()

                    scale_coord_map, mask = self.input_matrix_wpn(H * int_scale,W * int_scale, res_scale)
                    #position, mask = self.pos_matrix(H,W,self.args.scale[idx_scale])
                    #print(timer_test.toc())
                    mask = mask.to("cuda:0")
                    scale_coord_map = scale_coord_map.to("cuda:0")
                    timer_test.tic()
                    # print(mask.size())
                    # print(outH,outW)

                    sr = self.model(lr[0], scale_coord_map)
                    if isinstance(sr, list): sr = sr[-1]


                    
                    srDct = dct_2d(sr)
                    srDct = srDct[:, :, 0:int(outH), 0:int(outW)] * ((scale*scale)/16)
                    re_sr  = idct_2d(srDct)

                    re_sr= re_sr.contiguous().view(N, C,outH,outW)
                    re_sr = utility.quantize(re_sr, self.opt.rgb_range)
                    sr = re_sr

                    timer_test.hold()
                    # sr = sr.contiguous().view(N,C,int(H*4),int(W*4))
                    # sr = utility.quantize(sr, self.opt.rgb_range)

                    # R = sr[0, 0, :, :].type(torch.int)
                    # G = sr[0, 1, :, :].type(torch.int)
                    # B = sr[0, 2, :, :].type(torch.int)

                    # DR = dct_2d(R)
                    # DG = dct_2d(G)
                    # DB = dct_2d(B)

                    # CR = DR[:int(outH), :int(outW)] * ((scale*scale)/16)
                    # CG = DG[:int(outH), :int(outW)] * ((scale*scale)/16)
                    # CB = DB[:int(outH), :int(outW)] * ((scale*scale)/16)


                    # IR  = idct_2d(CR).type(torch.int)
                    # IB  = idct_2d(CG).type(torch.int)
                    # IG  = idct_2d(CB).type(torch.int)

                    # sr_dct = torch.stack((IR, IB, IG), dim=2).permute(2,0,1).unsqueeze(0)

                    if not no_eval:
                        psnr = utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                   
                        # hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                        # sr_numpy = sr[0].cpu().numpy().transpose(1, 2, 0)
                        # simm = utility.SSIM(hr_numpy, sr_numpy)
                        # eval_simm += simm

                        eval_psnr +=psnr

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                # eval_simm = eval_simm / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                # print('SIMM:',eval_simm)



        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')
        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
        