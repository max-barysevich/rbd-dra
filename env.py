import gym
from gym import spaces

import tifffile

import numpy as np

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from einops import rearrange

import os
import random

#%%

class DiscreteIllumControlEnv(gym.Env):

    def __init__(self,
                 dpath,
                 device='cpu',
                 render_mode=None,
                 img_dim=128,
                 episode_length=None,
                 max_pauses_per_seq=3,
                 pause_dur=[5,30],
                 termination_psnr=8.2,
                 num_seqs=5,
                 frames_per_seq=None,
                 crossfade_over=7,
                 augment_scale=.2,
                 augment_shear=45,
                 augment_brightness=.5,
                 augment_contrast=[10,100],
                 salt_prob=.002,
                 salt_sigma=[.3,.8],
                 oof_sigma=[3,10],
                 oof_brightness=[0.,.4],
                 illum_ratio=[10,100],
                 exp_t_ratio=[5,30],
                 default_ise_product=[100,500],
                 bleach_exp_fudge=[.01,.02],
                 gauss_mean_frac=[.005,.02],
                 charbonnier_epsilon=1e-3,
                 reward_bonus=0,
                 render_fps=25):
        
        # set device
        self.device = device
        
        # setting up dataset loading
        self.dpath = dpath

        self.seqs = os.listdir(dpath)
        self.frame_seqs = []

        for seq in self.seqs:
            frame_list = []
            for frame in os.listdir(os.path.join(dpath,seq)):
                frame_full = os.path.join(dpath,seq,frame)
                frame_list.append(frame_full)
            self.frame_seqs.append(frame_list)

        # setting up creating video sequences for the env
        self.img_dim = img_dim
        self.num_seqs = num_seqs
        self.frames_per_seq = frames_per_seq
        if frames_per_seq is not None:
            self.min_frames_per_seq = frames_per_seq[0]
            self.max_frames_per_seq = frames_per_seq[1]
        self.crossfade_over = crossfade_over
        
        if episode_length is not None:
            self.min_ep_length = episode_length[0]
            self.max_ep_length = episode_length[1]
            self.use_whole_sequence = False
        else:
            self.use_whole_sequence = True
            
        self.termination_psnr = termination_psnr

        self.frames_ep = None
        self.ep_length = None
        self.crossfade_at = None
        self.crossfade_with_i = None
        self.oof_source = None

        self.t_max = None
        
        # setting up pausing
        self.max_pauses_per_seq = max_pauses_per_seq
        self.min_pause_dur = pause_dur[0]
        self.max_pause_dur = pause_dur[1]

        # setting up crossfading
        self.crossfade_source = None
        self.crossfade_mul = None

        # setting up augmentation parameters
        self.scale = augment_scale
        self.shear = augment_shear
        self.brightness = augment_brightness
        self.contrast_min = augment_contrast[0]
        self.contrast_max = augment_contrast[1]
        self.salt_prob = salt_prob
        self.salt_sigma_min = salt_sigma[0]
        self.salt_sigma_max = salt_sigma[1]

        self.augmentor = None
        
        self.small_transform = transforms.RandomAffine(
            degrees=.5,
            translate=(.002,.002),
            scale=(.995,1.005),
            shear=(-.005,.005,-.005,.005)
            )

        self.salt_ = None
        self.salt_ones = None
        self.salt_zeros = None
        self.salt_sigma = None
        self.salt_flow = transforms.RandomAffine(degrees=10,
                                                 translate=(.05,.05),
                                                 scale=(.9,1.1),
                                                 shear=(-.05,.05,-.05,.05))
        
        self.oof_source = None
        self.oof = None
        self.min_oof_sigma = oof_sigma[0]
        self.max_oof_sigma = oof_sigma[1]
        self.min_oof_brightness = oof_brightness[0]
        self.max_oof_brightness = oof_brightness[1]
        self.oof_transform = transforms.RandomAffine(
            degrees=5,
            translate=(.01,.01),
            scale=(.99,1.01),
            shear=(-.01,.01,-.01,.01)
            )
        self.oof_blur = None

        # setting up possible illumination ratios
        self.illum_ratio_min = illum_ratio[0]
        self.illum_ratio_max = illum_ratio[1]
        self.illum_ratio = None

        # setting up possible exposure time ratios
        self.exp_ratio_min = exp_t_ratio[0]
        self.exp_ratio_max = exp_t_ratio[1]

        # setting up bleaching
        self.t = None
        # illumination x staining x exposure time for poisson multiplier
        self.min_ise = default_ise_product[0]
        self.max_ise = default_ise_product[1]
        self.min_fudge = bleach_exp_fudge[0]
        self.max_fudge = bleach_exp_fudge[1]

        self.ise_0 = None
        self.ise = None
        self.exp_sum = None
        self.fudge = None

        # setting up detection noise
        self.min_gauss_mean = gauss_mean_frac[0]
        self.max_gauss_mean = gauss_mean_frac[1]

        self.gauss_mean = None
        self.gauss_std = None

        # setting up reward parameters
        self.charbonnier_eps = charbonnier_epsilon
        self.min_loss = charbonnier_epsilon
        self.reward_bonus = reward_bonus # makes some rewards positive

        # setting up reference placeholder
        self.ref = None

        # setting up storing previous image
        self.prev_img = None

        # setting up observation and action spaces

        self.observation_space = spaces.Dict(
            {
                'ref': spaces.Box(0.,
                                  1.,
                                  shape=(1,img_dim,img_dim),
                                  dtype=np.float32),
                'fov': spaces.Box(0.,
                                  1.,
                                  shape=(1,img_dim,img_dim),
                                  dtype=np.float32),
                't': spaces.Box(0,
                                500,
                                shape=(),
                                dtype=np.int16)
                })

        self.action_space = spaces.Dict(
            {
                'denoiser_out': spaces.Box(0.,
                                           1.,
                                           shape=(1,img_dim,img_dim),
                                           dtype=np.float32),
                'action': spaces.Discrete(2)
                })

        # setting up rendering
        self.display_window = None
        self.display = None
        self.render_frame_dur = 1000 / render_fps

        self.render_clean = None
        self.render_noisy = None
        self.render_proc = None
    
    def seed(self,seed):
        random.seed(seed)
        np.random.seed(seed)

    def _create_sequence(self):

        # randomly pick n sequences from self.frame_seqs
        seqs = random.sample(self.frame_seqs,self.num_seqs)
        
        # pausing goes here
        if self.max_pauses_per_seq:
            for seq in seqs:
                n_pauses = np.random.randint(1,self.max_pauses_per_seq+1)
                n_pauses = len(seq) if n_pauses > len(seq) else n_pauses
                # sample n indices to pause at
                pause_inds = sorted(random.sample(range(len(seq)),n_pauses),
                                    reverse=True)
                # duplicate: make lists
                pause_durs = [
                    np.random.randint(self.min_pause_dur,
                                      self.max_pause_dur) for _ in pause_inds
                    ]
                
                pause_seqs = [
                    [
                        seq[pause_ind] for _ in range(pause_dur)
                        ] for pause_ind, pause_dur in zip(pause_inds,pause_durs)
                    ]
                # insert
                for pause_ind,pause_seq in zip(pause_inds,pause_seqs):
                    seq[pause_ind:pause_ind] = pause_seq

        self.frames_ep = []
        self.crossfade_at = []
        crossfade_i = [0]

        if self.frames_per_seq is not None:

            for i, seq in enumerate(seqs):
                #print(f'len(seq) at seqs[{i}] = {len(seq)}')
                frames_per_seq = np.random.randint(self.min_frames_per_seq,
                                                   self.max_frames_per_seq)
                
                start_i = np.random.randint(0,
                                            np.clip(len(seq)-frames_per_seq,
                                                    1,
                                                    None))
                frames = seq[start_i:start_i + frames_per_seq]
                [self.frames_ep.append(f) for f in frames]
                crossfade_i.append(crossfade_i[-1] +
                                   frames_per_seq)

        else:
            for seq in seqs:
                [self.frames_ep.append(f) for f in seq]
                crossfade_i.append(crossfade_i[-1] + len(seq))

        self.t_max = len(self.frames_ep)-1
        
        self.t = 0
        
        if not self.use_whole_sequence:
            
            self.ep_length = np.random.randint(self.min_ep_length,
                                               self.max_ep_length)
    
            self.ep_length = np.clip(np.random.randint(self.min_ep_length,
                                                       self.max_ep_length),
                                     0,
                                     self.t_max-1)
    
            self.t = np.random.randint(0,self.t_max-self.ep_length)
    
            self.t_max = self.t + self.ep_length - 1
        
        # create out-of-focus light source (first frame of each sequence)
        self.oof_source = [self.t] + [f for f in crossfade_i if f >= self.t]
    
        # disable crossfading outside of sequence
        crossfade_i = [f for f in crossfade_i if f > self.t] # first frame of each sequence after first
        self.crossfade_with_i = [i-1 for i in crossfade_i] # final frame of each sequence including first

        for i in crossfade_i:
            for j in range(self.crossfade_over):
                self.crossfade_at.append(i+j)
    
    @torch.no_grad()
    def _crossfade(self,im):

        if self.t in self.crossfade_with_i:
            self.crossfade_source = im

        if self.t in self.crossfade_at:
            self.crossfade_mul += 1
            #im = (self.crossfade_source + im) / (1 + self.crossfade_mul)
            im = (self.crossfade_source + self.crossfade_mul * im) / (1 + self.crossfade_mul)
            # try maximum projection instead
        else:
            self.crossfade_mul = 0

        return im

    @torch.no_grad()
    def _create_augmentor(self):

        # rotate
        angle = np.random.uniform(-180,180)

        # scale
        scale = 1 + np.random.uniform(-self.scale,self.scale)

        # shear
        shear = (
            np.random.uniform(-self.shear,self.shear),
            np.random.uniform(-self.shear,self.shear)
            )

        # flip
        hflip = np.random.randint(2)
        vflip = np.random.randint(2)

        # invert
        invert = np.random.randint(2)

        # adjust brightness
        brightness_factor = 1 + np.random.uniform(-self.brightness,
                                                  self.brightness)

        # increase contrast and clip
        contrast_factor = np.random.uniform(self.contrast_min,
                                            self.contrast_max)

        # return the augmentation function
        self.augmentor = lambda im: self._augment(im,
                                                  angle,
                                                  scale,
                                                  shear,
                                                  hflip,
                                                  vflip,
                                                  invert,
                                                  brightness_factor,
                                                  contrast_factor)

    @torch.no_grad()
    def _add_salt(self,
                  im):
        
        salt = TF.gaussian_blur(self.salt,5,self.salt_sigma)

        return torch.clamp(im+salt,min=0,max=1)
    
    @torch.no_grad()
    def _create_salt(self):
        
        self.salt_ones = torch.ones(1,self.img_dim,self.img_dim).to(self.device)
        self.salt_zeros = torch.zeros(1,self.img_dim,self.img_dim).to(self.device)
        self.salt_sigma = np.random.uniform(self.salt_sigma_min,
                                            self.salt_sigma_max)
        
        salt = torch.rand(1,self.img_dim,self.img_dim).to(self.device)
        salt = torch.where(salt<self.salt_prob,
                           self.salt_ones,
                           self.salt_zeros)
        self.salt = salt
    
    @torch.no_grad()
    def _update_salt(self):
        salt = rearrange(self.salt,
                          '... (nh wh) (nw ww) -> ... (nh nw) wh ww',
                          nh=4,nw=4)
        salt = torch.split(salt,1,dim=-3)
        
        salt = [self.salt_flow(s) for s in salt]
        salt = torch.cat(salt,dim=-3)
        salt = rearrange(salt,
                         '... (nh nw) wh ww -> ... (nh wh) (nw ww)',
                         nh=4,nw=4)
        self.salt = salt

    @torch.no_grad()
    def _augment(self,
                 im,
                 angle,
                 scale,
                 shear,
                 hflip,
                 vflip,
                 invert,
                 brightness_factor,
                 contrast_factor):

        im = TF.resize(im,
                       self.img_dim)

        if hflip:
            im = TF.hflip(im)
        if vflip:
            im = TF.vflip(im)
        if invert:
            im = TF.invert(im)

        im = TF.affine(im,
                       angle,
                       (0,0),
                       scale,
                       shear)

        im = TF.adjust_brightness(im,
                                  brightness_factor)

        im = im.mean() + contrast_factor*(im-im.mean())

        im = torch.clamp(im,min=0)/im.max()
        
        #im = self.small_transform(im)

        im = self._add_salt(im)

        return im

    @torch.no_grad()
    def _bleach(self,
                im,
                ir,
                tr):

        # ise_i = ise_0 * ir_i * tr_i * exp(-fudge*sum(ir_j*tr_j for j=0:i-1))

        self._update_ise(ir,tr)

        # generate poisson image
        imn = torch.poisson(im*self.ise)

        # gaussian noise mean should be fixed before normalisation
        # to ensure consistent detection noise
        n = torch.normal(self.gauss_mean*torch.ones_like(imn),
                         self.gauss_std*torch.ones_like(imn)).to(self.device)

        imn = torch.clamp(imn + n,min=0)

        return imn/imn.max()

    def _update_ise(self,ir,tr):

        # update illum x staining x exposure product
        self.ise = self.ise_0 * ir * tr * np.exp(-self.fudge*self.exp_sum)

        # update exponent sum for next iteration
        self.exp_sum += ir * tr

    def _charbonnier_loss(self,gt,y):
        return (((gt-y)**2).mean() + self.charbonnier_eps**2)**.5
    
    def _psnr(self,img,imn):
        mse = torch.mean((img-imn)**2)
        return 10*torch.log10((img.max()**2) / mse)

    @torch.no_grad()
    def reset(self,
              seed=None,
              options=None):
        super().reset(seed=seed)

        # set up loading a new video sequence
        self._create_sequence()
        # self.t is set by above

        # set up cross-fading
        self.crossfade_mul = 0
        self.crossfade_source = None

        # reset photobleaching
        self.ise_0 = np.random.uniform(self.min_ise,self.max_ise)
        self.ise = None
        self.fudge = np.random.uniform(self.min_fudge,self.max_fudge)
        self.exp_sum = 0

        # set new augmentation parameters
        self._create_salt()
        self._create_augmentor()
        
        self.oof = None
        oof_sigma = np.random.uniform(self.min_oof_sigma,
                                      self.max_oof_sigma)
        oof_brightness = np.random.uniform(self.min_oof_brightness,
                                           self.max_oof_brightness)
        self.oof_blur = lambda x: TF.adjust_brightness(
            TF.gaussian_blur(x,11,oof_sigma),
            oof_brightness
            )

        # set new illumination change ratio
        self.illum_ratio = np.random.uniform(self.illum_ratio_min,
                                             self.illum_ratio_max)

        # set new gaussian noise settings
        self.gauss_mean = self.ise_0 * np.random.uniform(self.min_gauss_mean,
                                                         self.max_gauss_mean)
        self.gauss_std = self.gauss_mean / 3 # to ensure few negative values

        # reference placeholder
        self.ref = None
        self.prev_img = torch.zeros(1,self.img_dim,self.img_dim).to(self.device)
        
        action = {'denoiser_output': self.prev_img.cpu().numpy(),
                  'action': 1}

        obs, *_, info = self.step(action)

        return obs, info

    @torch.no_grad()
    def step(self,action):

        # action is a tuple of (denoiser_output, agent_output)

        #denoiser_output = action['denoiser_output'].to(self.device)
        denoiser_output = torch.tensor(action['denoiser_output']).to(self.device)
        action = action['action']

        # decode action, produce ir_i and tr_i
        ir_i = [1 if action==1 else 1/self.illum_ratio][0]
        tr_i = 1 # exposure time not controlled yet

        # reward = neg reconstruction loss
        reward = (self.min_loss -
                  self._charbonnier_loss(self.prev_img,denoiser_output) +
                  self.reward_bonus).cpu().numpy()

        # get next frame
        im = read_image(self.frames_ep[self.t],
                        mode=ImageReadMode.GRAY).to(self.device)/255

        # augment the image to make it more similar to FM data
        img = self.augmentor(im)
        
        # move point sources
        self._update_salt()
        if self.salt.max() < 1e-3:
            self._create_salt()
        
        # update oof
        self.oof = self.oof_transform(
            img if self.t in self.oof_source else self.oof
            )
        
        oof = self.oof_blur(self.oof)
        
        img = img + oof
        img = torch.clamp(img/img.max(),min=0)

        # cross-fade
        img = self._crossfade(img)
        
        # update prev_img with new gt
        self.prev_img = img

        # bleach and detect the frame according to bleaching clock and action
        imn = self._bleach(img,ir_i,tr_i)

        # possibly update self.ref
        self.ref = [imn if action==1 else self.ref][0]

        obs = {'ref': self.ref.cpu().numpy(),
               'fov': imn.cpu().numpy(),
               't': self.t}

        terminated = self.t == self.t_max
        if not terminated and self.termination_psnr is not None:
            terminated = (self._psnr(img,imn) < self.termination_psnr).item()

        truncated = False

        info = {
            't': self.t
            }

        self.t += 1

        # update render
        self.render_clean = img # tensor
        self.render_noisy = imn # tensor
        self.render_proc = denoiser_output # array

        return obs, reward, terminated, truncated, info
    
    def _prep_render(self,im):

        if im is not None:
            im = (im*255).type(torch.uint8).cpu().numpy()[0]

        return im
    
    def fake_render(self):

        _ = self.reset()

        self.count = 1

        frame_list = []
        terminated = False
        
        action1 = {'denoiser_output': self.prev_img.cpu().numpy(),
                   'action': 1}
        
        action2 = {'denoiser_output': self.prev_img.cpu().numpy(),
                   'action': 0}

        while not terminated:
            frame_list.append(self._prep_render(self.render_noisy))
            
            if self.count % 10 == 0:
                _,_,terminated,*_ = self.step(action1)
            else:
                _,_,terminated,*_ = self.step(action2)
            self.count += 1

        tiff = np.stack(frame_list)
        tifffile.imwrite('default_render.tif',tiff)
    
    def render(self,agent):
        
        obs, _ = self.reset()
        
        support = torch.linspace(-10,10,51)
        
        state = None
        
        count = 1
        
        terminated = False
        
        psnr_dra = []
        psnr_def = []
        
        ssim_dra = []
        ssim_def = []
        
        while not terminated:
            
            print(f'Frame {count}.')
            
            ref = rearrange(obs['ref'],'... -> 1 ...')
            imn = rearrange(obs['fov'],'... -> 1 ...')
            obs = {'ref':ref,'fov':imn}
            
            with torch.no_grad():
                action, state = agent(obs,state)
            
            logits = action['logits']
            act = (logits.cpu()*support).sum(2).max(dim=1)[1].item()
            
            # compare action with render_clean
            denoiser_output = action['denoiser_output'].cpu()
            
            psnr_dra.append(self._psnr(self.render_clean,denoiser_output))
            ssim_dra.append(None)
            
            psnr_def.append(self._psnr(self.render_clean,self.render_noisy))
            ssim_def.append(None)
            
            action = {'denoiser_output': denoiser_output.numpy(),
                      'action': act}
            
            obs, _, terminated, *_ = self.step(action)
            
            count += 1
        
        return psnr_dra, psnr_def
    
    def render_non_rl(self,denoiser,dra=True):
        
        obs, _ = self.reset()
        
        count = 1
        frame_list = []
        
        terminated = False
        
        action1 = {'denoiser_output': self.prev_img.numpy(),
                   'action': 1}
        
        if dra:
            action2 = {'denoiser_output': self.prev_img.numpy(),
                       'action': 0}
            name = 'dra_enabled'
        else:
            action2 = {'denoiser_output': self.prev_img.numpy(),
                       'action': 1}
            name = 'dra_disabled'
        
        psnr = []
        
        while not terminated:
            
            print(f'Frame {count}.')
            
            if dra:
            
                ref = torch.tensor(rearrange(obs['ref'],'... -> 1 ...'),
                                   device='cuda')
                imn = torch.tensor(rearrange(obs['fov'],'... -> 1 ...'),
                                   device='cuda')
                denoiser_in = [ref,imn]
                
                with torch.no_grad():
                    denoiser_out = denoiser(denoiser_in).cpu()
                
                psnr.append(self._charbonnier_loss(self.render_clean,denoiser_out))
                frame_list.append(self._prep_render(denoiser_out))
            
            else:
            
                psnr.append(self._charbonnier_loss(self.render_clean,self.render_noisy))
                frame_list.append(self._prep_render(self.render_noisy))
            
            if count % 10 == 0:
                obs, _, terminated, *_ = self.step(action1)
            else:
                obs, _, terminated, *_ = self.step(action2)
            
            count += 1
        
        tiff = np.stack(frame_list)
        tifffile.imwrite(name+'.tif',tiff)
        
        return psnr