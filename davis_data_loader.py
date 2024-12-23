# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from einops import rearrange

#%%

class dataset(Dataset):

    def __init__(self,
                 file_list,
                 img_dim=256,
                 max_angle=180,
                 translate=(0.,0.),
                 scale=(.9,1.1),
                 shear=(-.2,.2,-.2,.2),
                 brightness = .3,
                 ref_angle=10,
                 ref_translate=(.05,.05),
                 ref_scale=(.99,1.01),
                 ref_shear=(-.01,.01,-.01,.01),
                 point_angle=5,
                 point_translate=(.1,.1),
                 point_scale=(.9,1.1),
                 point_shear=(-.1,.1,-.1,.1),
                 contrast = [10,100],
                 point_prob = .05,
                 point_sigma = [.3,.8],
                 oof_prob = .4,
                 oof_kernel_size = 11,
                 oof_blur_sigma = [3,10],
                 oof_brightness = [0.,.4],
                 exposure_range=[1,50],
                 exposure_ratio=[10,100],
                 ref_exposure_range=[50,500],
                 gauss_max_mean=10,
                 gauss_max_std=5):
        self.file_list = file_list
        self.img_dim = img_dim
        self.contrast = contrast
        self.point_prob = point_prob
        self.point_sigma = point_sigma
        self.exp_min = exposure_range[0]
        self.exp_max = exposure_range[1]
        self.exp_ratio_min = exposure_ratio[0]
        self.exp_ratio_max = exposure_ratio[1]
        self.ref_exp_range = ref_exposure_range
        self.gn_mean = gauss_max_mean
        self.gn_std = gauss_max_std

        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=max_angle,
                                    translate=translate,
                                    scale=scale,
                                    shear=shear),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomInvert(),
            transforms.RandomCrop((img_dim,img_dim))
            ])

        self.ref_addtl_transform = transforms.RandomAffine(degrees=ref_angle,
                                                           translate=ref_translate,
                                                           scale=ref_scale,
                                                           shear=ref_shear)

        self.point_flow = transforms.RandomAffine(degrees=point_angle,
                                                  translate=point_translate,
                                                  shear=point_shear)

        self.gt_point_scale = transforms.RandomAffine(degrees=0,
                                                      scale=point_scale)

        self.oof_transform = transforms.RandomApply(
            [transforms.Compose([
                transforms.RandomAffine(degrees=max_angle,
                                        translate=translate,
                                        scale=scale,
                                        shear=shear),
                transforms.GaussianBlur(oof_kernel_size,
                                        sigma=oof_blur_sigma),
                transforms.ColorJitter(brightness=oof_brightness)
                ])],
            p=oof_prob
            )

    def adjust_contrast(self,x,ratio):
        x = x.mean() + ratio*(x-x.mean())
        x = torch.clamp(x,min=0)/x.max()
        if torch.isnan(x.mean()):
            x = torch.zeros_like(x)
        return x

    def points(self,salt_prob,sigma):
        c = torch.rand([1,self.img_dim,self.img_dim])
        ones = torch.ones([1,self.img_dim,self.img_dim])
        zeros = torch.zeros([1,self.img_dim,self.img_dim])
        salt_ = torch.where(c<salt_prob,ones,zeros)

        salt_ = rearrange(salt_,
                          '... (nh wh) (nw ww) -> ... (nh nw) wh ww',
                          nh=4,nw=4)
        salt_ = torch.split(salt_,1,dim=-3)

        salt = [self.gt_point_scale(s) for s in salt_]
        salt = torch.cat(salt,dim=-3)
        salt = rearrange(salt,
                         '... (nh nw) wh ww -> ... (nh wh) (nw ww)',
                         nh=4,nw=4)

        salt_ref = [self.point_flow(s) for s in salt_]
        salt_ref = torch.cat(salt_ref,dim=-3)
        salt_ref = rearrange(salt_ref,
                             '... (nh nw) wh ww -> ... (nh wh) (nw ww)',
                             nh=4,nw=4)

        return [transforms.functional.gaussian_blur(salt,5,sigma=sigma),
                transforms.functional.gaussian_blur(salt_ref,5,sigma=sigma)]

    def add_noise(self,imn):

        # Poisson
        self.exposure = self.exp_min + torch.rand(())*(self.exp_max-self.exp_min)
        exposure = self.exposure
        imn = torch.poisson(imn*exposure) # default ise = [100,200]

        # Gaussian
        mean = torch.rand(())*self.gn_mean
        std = torch.rand(())*self.gn_std
        n = torch.normal(mean*torch.ones([self.img_dim,self.img_dim]),
                         std*torch.ones([self.img_dim,self.img_dim]))
        imn = imn + n
        imn = torch.clamp(imn/imn.max(),min=0)
        return imn

    def add_ref_noise(self,ref):

        exposure_ratio = self.exp_ratio_min + torch.rand(())*\
            (self.exp_ratio_max - self.exp_ratio_min)

        exposure = self.exposure * exposure_ratio

        exposure = torch.clip(exposure,
                              self.ref_exp_range[0],
                              self.ref_exp_range[1])

        ref = torch.poisson(ref*exposure) # default ise = [100,200]

        # Gaussian
        mean = torch.rand(())*self.gn_mean
        std = torch.rand(())*self.gn_std
        n = torch.normal(mean*torch.ones([self.img_dim,self.img_dim]),
                         std*torch.ones([self.img_dim,self.img_dim]))
        ref = ref + n
        ref = torch.clamp(ref/ref.max(),min=0)
        return ref

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self,idx):

        x = read_image(self.file_list[idx])/255

        x = self.transform(x)

        contrast_ratio = self.contrast[0] + torch.rand(())*(self.contrast[1]
                                                            -self.contrast[0])
        x = self.adjust_contrast(x,contrast_ratio)

        # split here
        ref,_,gt = x.split(1)

        ref = self.ref_addtl_transform(ref)

        points_sigma = self.point_sigma[0] + torch.rand(())*(self.point_sigma[1]-
                                                             self.point_sigma[0])
        point_prob = torch.rand(())*self.point_prob

        points_gt,points_ref = self.points(point_prob.item(),points_sigma.item())

        ref = ref + points_ref
        gt = gt + points_gt

        # add out-of-focus light
        oof_n = self.oof_transform(gt)
        oof_ref = self.oof_transform(ref)

        ref = ref + oof_ref
        n = gt + oof_n

        ref = torch.clamp(ref/ref.max(),min=0)
        n = torch.clamp(n/n.max(),min=0)
        gt = torch.clamp(gt/gt.max(),min=0)

        n = self.add_noise(n)
        ref = self.add_ref_noise(ref)

        return ((ref,n),gt)
