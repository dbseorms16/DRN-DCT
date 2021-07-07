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
    def input_matrix_wpn(self,inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale*inH), int(scale*inW)

        #### mask records which pixel is invalid, 1 valid or o invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
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

        return pos_mat,mask_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

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
            if splitname[1] != 'SRCNN':
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
            sr = self.model(lr[0])

            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[(i-1) - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

            # compute primary loss
            ##여기서 sr사이즈 hr사이즈
            loss_primary = self.loss(sr[-1], hr)
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])

            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])

            # compute total loss
            loss = loss_primary + self.opt.dual_weight * loss_dual
            
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


                    N,C,H,W = lr[0].size()
                    outH,outW = int(H*scale),int(W*scale)
                    timer_test.tic()
                    
                    if self.opt.arbit == True :
                        scale_coord_map, mask = self.input_matrix_wpn(H,W, scale)
                        scale_coord_map = scale_coord_map.to('cuda')
                        sr = self.model(lr[0], scale_coord_map)
                        if isinstance(sr, list): sr = sr[-1]
                        sr = torch.masked_select(sr,mask.to('cuda'))
                    else :
                        sr = self.model(lr[0])
                        if isinstance(sr, list): sr = sr[-1]
                    


                    sr = sr.contiguous().view(N, C,outH,outW)
                    sr = utility.quantize(sr, self.opt.rgb_range)

                    timer_test.hold()
               

                    if not no_eval:
                        psnr = utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                   
                    #     # hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                    #     # sr_numpy = sr[0].cpu().numpy().transpose(1, 2, 0)
                    #     # simm = utility.SSIM(hr_numpy, sr_numpy)
                    #     # eval_simm += simm

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
        