# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

#%%

class LeFF(nn.Module):
    def __init__(self,
                 filters_inner=64,
                 filters_outer=16,
                 img_dim=128):
        super(LeFF,self).__init__()

        self.lp1 = nn.Linear(in_features=filters_outer,
                             out_features=filters_inner)

        self.gelu1 = nn.GELU()

        self.rearrange1 = Rearrange('b (h w) c -> b c h w',
                                    h=img_dim)

        self.conv = nn.Conv2d(in_channels=filters_inner,
                              out_channels=filters_inner,
                              kernel_size=3,
                              padding='same')

        self.gelu2 = nn.GELU()

        self.rearrange2 = Rearrange('b c h w -> b (h w) c')

        self.lp2 = nn.Linear(in_features=filters_inner,
                             out_features=filters_outer)

        self.gelu3 = nn.GELU()

    def forward(self,x):

        x = self.lp1(x)
        x = self.gelu1(x)
        x = self.rearrange1(x)
        x = self.conv(x)
        x = self.gelu2(x)
        x = self.rearrange2(x)
        x = self.lp2(x)
        x = self.gelu3(x)

        return x

class LeWin(nn.Module):
    def __init__(self,
                 proj_patch_dim=16,
                 shifted_windows=False,
                 include_modulator=True,
                 attn_heads=8,
                 attn_dim=32,
                 dropout_rate=.1,
                 leff_filters=32,
                 leff_filters_out=16,
                 fmap_dim=64):
        super(LeWin,self).__init__()
        self.proj_patch_dim = proj_patch_dim
        self.attn_heads = attn_heads
        self.attn_dim = attn_dim
        self.dropout_rate = dropout_rate
        self.leff_filters = leff_filters
        self.leff_filters_out = leff_filters_out
        self.fmap_dim = fmap_dim

        self.norm1 = nn.LayerNorm(leff_filters_out)

        self.window = Rearrange('b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c',
                                nh=fmap_dim//proj_patch_dim,
                                nw=fmap_dim//proj_patch_dim,
                                wh=proj_patch_dim,
                                ww=proj_patch_dim)

        if include_modulator:

            self.modulator = nn.Embedding(proj_patch_dim*proj_patch_dim,
                                          leff_filters_out)
            self.add_modulator = lambda x: x + self.modulator.weight

        else:
            self.add_modulator = lambda x: x

        self.window_reverse = Rearrange(
            '(b nh nw) (wh ww) c -> b (nh wh nw ww) c',
            nh=self.fmap_dim//self.proj_patch_dim,
            nw=self.fmap_dim//self.proj_patch_dim,
            wh=self.proj_patch_dim,
            ww=self.proj_patch_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(leff_filters_out)

        self.leff = LeFF(filters_inner=leff_filters,
                         filters_outer=leff_filters_out,
                         img_dim=fmap_dim)

        # positional encoding

        self.rpbt = nn.Parameter(
            torch.zeros((2*proj_patch_dim-1)**2,
                        attn_heads))

        x = torch.arange(0,proj_patch_dim)
        y = torch.arange(0,proj_patch_dim)
        c = torch.stack(torch.meshgrid(x,y))
        f = rearrange(c,'n x y -> n (x y)')
        r = f[:,:,None]-f[:,None,:]
        r0 = r[0]
        r1 = r[1]
        r0 = (r0 + proj_patch_dim - 1)*(2*proj_patch_dim - 1)
        r1 = r1 + proj_patch_dim - 1
        rpi = r0 + r1
        self.register_buffer('rpi',rpi)

        # shift mask - should be in forward()
        if shifted_windows:

            shift_mask = torch.zeros(1,self.fmap_dim,self.fmap_dim,1)
            #shift_mask = torch.zeros(1,256,256,1)

            h_slices = (slice(0, -self.proj_patch_dim),
                        slice(-self.proj_patch_dim, -self.proj_patch_dim//2),
                        slice(-self.proj_patch_dim//2, None))
            w_slices = (slice(0, -self.proj_patch_dim),
                        slice(-self.proj_patch_dim, -self.proj_patch_dim//2),
                        slice(-self.proj_patch_dim//2, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1

            shift_mask_windows = rearrange(shift_mask,
                                           'b (nh wh) (nw ww) c -> (b nh nw c) (wh ww)',
                                           wh=self.proj_patch_dim,
                                           ww=self.proj_patch_dim)

            shift_attn_mask = shift_mask_windows.unsqueeze(1) - \
                shift_mask_windows.unsqueeze(2)
            shift_attn_mask = shift_attn_mask.masked_fill(
                shift_attn_mask != 0,
                float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            #shift_attn_mask = rearrange(shift_attn_mask,'b ... -> b 1 ...')
            shift_attn_mask = rearrange(shift_attn_mask,
                                        '(b nh nw) ... -> b (nh nw) 1 ...',
                                        nh=self.fmap_dim//self.proj_patch_dim,
                                        nw=self.fmap_dim//self.proj_patch_dim) # b=1
            #self.shift_attn_mask = shift_attn_mask
            self.register_buffer('shift_attn_mask',shift_attn_mask,persistent=False)

            self.add_shift_mask = lambda x: x + self.shift_attn_mask

            self.cyclic_shift = lambda x: torch.roll(
                x,
                shifts=[-self.proj_patch_dim//2,-self.proj_patch_dim//2],
                dims=[1,2])

            self.rev_cyclic_shift = lambda x: torch.roll(
                x,
                shifts=[self.proj_patch_dim//2,self.proj_patch_dim//2],
                dims=[1,2])

        else:
            self.add_shift_mask = lambda x: x
            self.cyclic_shift =lambda x: x
            self.rev_cyclic_shift =lambda x: x

        # attention

        self.attn_qkv = nn.Linear(leff_filters_out,
                                  3*attn_heads*attn_dim)

        self.attn_rev = nn.Linear(attn_heads*attn_dim,
                                  leff_filters_out)

    def forward(self,x):

        # input size: b (h w) c

        shortcut = x

        x = self.norm1(x)
        x = rearrange(x,'b (h w) c -> b h w c',h=self.fmap_dim)

        x = self.cyclic_shift(x)

        x = self.window(x)

        x = self.add_modulator(x)

        #x = self.attention(x)
        ### attention (self,x)
        qkv = self.attn_qkv(x)

        qkv = rearrange(qkv,'b n (i h a) -> i b h n a',
                        i=3,
                        h=self.attn_heads)

        q,k,v = torch.chunk(qkv,3)
        q,k,v = q[0], k[0], v[0]

        q = q * self.attn_dim ** -.5

        attn = (q @ k.transpose(-2,-1))

        rpb = torch.index_select(self.rpbt,
                                 0,
                                 self.rpi.view(-1))

        rpb = rearrange(rpb,
                        '(i j) h -> 1 h i j',
                        i=self.proj_patch_dim**2)

        attn = attn + rpb

        attn = rearrange(attn,'(b nh nw) ... -> b (nh nw) ...',
                         nh=self.fmap_dim//self.proj_patch_dim,
                         nw=self.fmap_dim//self.proj_patch_dim)
        attn = self.add_shift_mask(attn) # remember to add cyclic shift
        attn = rearrange(attn,'b (nh nw) ... -> (b nh nw) ...',
                         nh=self.fmap_dim//self.proj_patch_dim,
                         nw=self.fmap_dim//self.proj_patch_dim)

        attn = F.softmax(attn,dim=-1)

        attn = self.dropout(attn)

        x = attn @ v

        x = rearrange(x,'b h n a -> b n (h a)')

        x = self.attn_rev(x)
        # return x
        ###

        x = self.window_reverse(x)

        # reverse cyclic shift
        x = rearrange(x,'b (h w) c -> b h w c',h=self.fmap_dim)
        x = self.rev_cyclic_shift(x)
        x = rearrange(x,'b h w c -> b (h w) c')

        x = self.dropout(x)

        x_res = shortcut + x

        x = self.norm2(x_res)
        x = self.leff(x)
        x = self.dropout(x)

        return x + x_res

class Uformer(nn.Module):

    def __init__(self,
                 use_ref = True,
                 img_dim=64,
                 img_ch=1,
                 out_ch=1,
                 proj_dim=16,
                 proj_kernel=3,
                 proj_patch_dim=8,
                 attn_heads=[1,2,4,8],
                 attn_dim=32,
                 dropout_rate=.1,
                 leff_filters=64,
                 n=[2,2,2,2]):
        super(Uformer,self).__init__()
        self.img_dim = img_dim
        self.img_ch = img_ch
        self.out_ch = out_ch
        self.proj_dim = proj_dim
        self.proj_patch_dim = proj_patch_dim
        self.attn_heads = attn_heads
        self.attn_dim = attn_dim
        self.dropout_rate = dropout_rate
        self.leff_filters = leff_filters
        self.n = n

        self.concat = (lambda x: torch.cat([x[0],x[1]],dim=-3)) if use_ref else (lambda x: x[0])

        self.proj = nn.Sequential(
            nn.Conv2d(int(img_ch*2),
                      proj_dim,
                      proj_kernel,
                      padding='same'),
            nn.LeakyReLU(),
            Rearrange('b c h w -> b (h w) c'))

        self.proj_out = nn.Sequential(
            Rearrange('b (h w) c -> b c h w',
                      h=img_dim),
            nn.Conv2d(proj_dim*2,
                      out_ch,
                      proj_kernel,
                      padding='same')
            )

        lewin_down = []
        for i in range(len(n)-1):

            lewin_down.append(nn.Sequential(*[
                LeWin(proj_patch_dim=proj_patch_dim,
                      shifted_windows=False if j%2 == 0 else True,
                      include_modulator=False,
                      attn_heads=attn_heads[i],
                      attn_dim=attn_dim,
                      dropout_rate=dropout_rate,
                      leff_filters=leff_filters,
                      leff_filters_out=proj_dim*(2**i),
                      fmap_dim=img_dim//(2**i)) for j in range(n[i])
                ]))
        self.lewin_down = nn.ModuleList(lewin_down)

        lewin_up= []
        for i in reversed(range(len(n)-1)):

            lewin_up.append(nn.Sequential(*[
                LeWin(proj_patch_dim=proj_patch_dim,
                      shifted_windows=False if j%2 == 0 else True,
                      include_modulator=True,
                      attn_heads=attn_heads[i],
                      attn_dim=attn_dim,
                      dropout_rate=dropout_rate,
                      leff_filters=leff_filters,
                      leff_filters_out=proj_dim*2*(2**i),
                      fmap_dim=img_dim//(2**i)) for j in range(n[i])
                ]))
        self.lewin_up = nn.ModuleList(lewin_up)

        self.lewin_bottom = nn.Sequential(*[
            LeWin(proj_patch_dim=proj_patch_dim,
                  shifted_windows=False if j%2 == 0 else True,
                  include_modulator=False,
                  attn_heads=attn_heads[-1],
                  attn_dim=attn_dim,
                  dropout_rate=dropout_rate,
                  leff_filters=leff_filters,
                  leff_filters_out=proj_dim*(2**(len(n)-1)),
                  fmap_dim=img_dim//(2**(len(n)-1)) ) for j in range(n[-1])
            ])

        downsample = []
        for i in range(len(n)-1):

            downsample.append(nn.Sequential(
                Rearrange('b (h w) c -> b c h w',h=img_dim//(2**i)),
                nn.Conv2d(proj_dim*(2**i),
                          proj_dim*2*(2**i),
                          4,
                          stride=2,
                          padding=1),
                Rearrange('b c h w -> b (h w) c')
                ))
        self.downsample = nn.ModuleList(downsample)

        upsample = []

        upsample.append(nn.Sequential(
            Rearrange('b (h w) c -> b c h w',h=img_dim//(2**(len(n)-1))),
            nn.ConvTranspose2d(proj_dim*(2**(len(n)-1)),
                               proj_dim*(2**(len(n)-2)),
                               2,
                               stride=2),
            Rearrange('b c h w -> b (h w) c')
            ))

        for i in reversed(range(len(n)-2)):

            upsample.append(nn.Sequential(
                Rearrange('b (h w) c -> b c h w',h=img_dim//(2**(i+1))),
                nn.ConvTranspose2d(proj_dim*4*(2**i),
                                   proj_dim*(2**i),
                                   2,
                                   stride=2),
                Rearrange('b c h w -> b (h w) c')
                ))
        self.upsample = nn.ModuleList(upsample)

    def forward(self,inputs):

        ref_in, x_in = inputs

        #x = torch.cat([x_in,ref_in],dim=-3)
        x = self.concat([x_in,ref_in])
        x = self.proj(x)

        xlhr = [x]
        xllr = [x]

        for i in range(len(self.n)-1):

            xi = self.lewin_down[i](xllr[-1])

            xlhr.append(xi)

            xi = self.downsample[i](xi)

            xllr.append(xi)

        xb = self.lewin_bottom(xllr[-1])

        xllr2 = [xb]

        for i in range(len(self.n)-1):

            xi = self.upsample[i](xllr2[-1])

            xc = xlhr[-i-1]

            xi = torch.cat([xi,xc],dim=-1)

            xi = self.lewin_up[i](xi)

            xllr2.append(xi)

        x_out = self.proj_out(xllr2[-1])

        return x_out
