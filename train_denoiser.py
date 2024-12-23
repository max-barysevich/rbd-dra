# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import glob
import os
import datetime
from shutil import copy

from uformer import Uformer
from davis_data_loader import dataset
from utils import Charbonnier, psnr
from config import Config

#%%

resume_from = Config.RESUME_FROM
lr = Config.LEARNING_RATE
wd = Config.WEIGHT_DECAY
epochs = Config.EPOCHS
batch = Config.BATCH_SIZE
step_size = Config.SCHEDULING_STEP_SIZE
gamma = Config.SCHEDULING_GAMMA
charbonnier_epsilon = Config.CHARBONNIER_EPSILON
save_every = Config.BACKUP_RATE

img_dim = Config.IMG_DIM
img_ch = Config.IMG_CH
proj_dim = Config.PROJ_DIM
proj_patch_dim = Config.WINDOW_SIZE
attn_heads = Config.ATTN_HEADS
attn_dim = Config.ATTN_DIM
dropout_rate = Config.DROPOUT
leff_filters = Config.LEFF_FILTERS
n = Config.STAGES

max_angle = Config.MAX_ANGLE
translate = Config.TRANSLATE
scale = Config.SCALE
shear = Config.SHEAR
brightness = Config.BRIGHTNESS

ref_angle = 10
ref_translate = (.5,.5)
ref_scale = (.5,1.5)
ref_shear = (-.2,.2,-.2,.2)

contrast = Config.CONTRAST
point_prob = Config.POINT_PROB
point_sigma = Config.POINT_SIGMA
exposure_range = Config.EXPOSURE_RANGE
gauss_max_mean = Config.GAUSS_MAX_MEAN
gauss_max_std = Config.GAUSS_MAX_STD

#%%

subdir = datetime.datetime.now().strftime('run%Y%m%dT%H%M')

copy(Config.ROOT_DIR+'config.py',Config.CONFIG_DIR+'config_'+subdir+'.py')

if resume_from is None:
    log_dir = Config.LOG_DIR + subdir + '/'
    save_dir = Config.CHK_DIR + subdir + '/'
    os.mkdir(save_dir)
else:
    log_dir = Config.LOG_DIR + resume_from + '/'
    save_dir = Config.CHK_DIR + resume_from + '/'

writer = SummaryWriter(log_dir)

train_dir = Config.TRAIN_DIR
val_dir = Config.VAL_DIR

model = Uformer(img_dim=img_dim,
                img_ch=img_ch,
                proj_dim=proj_dim,
                proj_patch_dim=proj_patch_dim,
                attn_heads=attn_heads,
                attn_dim=attn_dim,
                dropout_rate=dropout_rate,
                leff_filters=leff_filters,
                n=n).cuda()

loss_fn = Charbonnier(epsilon=charbonnier_epsilon).cuda()
optimizer = optim.AdamW(model.parameters(),
                        lr=lr,
                        weight_decay=wd)

scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=step_size,
                                      gamma=gamma)
    
if resume_from is not None:
    
    last_checkpoint = os.listdir(Config.CHK_DIR + resume_from)[-1]
    
    checkpoint = torch.load(Config.CHK_DIR + resume_from + '/' + last_checkpoint)
    
    model.load_state_dict(checkpoint['model'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_epoch = checkpoint['epoch'] + 1
    
    for _ in range(start_epoch):
        scheduler.step()

else:
    
    start_epoch = 0

train_list = glob.glob(os.path.join(train_dir,'*.png'))
val_list = glob.glob(os.path.join(val_dir,'*.png'))

train_data = dataset(train_list,
                     img_dim=img_dim,
                     max_angle=max_angle,
                     translate=translate,
                     scale=scale,
                     shear=shear,
                     brightness = brightness,
                     ref_angle=ref_angle,
                     ref_translate=ref_translate,
                     ref_scale=ref_scale,
                     ref_shear=ref_shear,
                     contrast = contrast,
                     point_prob = point_prob,
                     point_sigma = point_sigma,
                     exposure_range=exposure_range,
                     gauss_max_mean=gauss_max_mean,
                     gauss_max_std=gauss_max_std)

val_data = dataset(val_list,
                   img_dim=img_dim,
                   max_angle=max_angle,
                   translate=translate,
                   scale=scale,
                   shear=shear,
                   brightness = brightness,
                   contrast = contrast,
                   point_prob = point_prob,
                   point_sigma = point_sigma,
                   exposure_range=exposure_range,
                   gauss_max_mean=gauss_max_mean,
                   gauss_max_std=gauss_max_std)

train_loader = DataLoader(train_data,
                          batch_size=batch,
                          shuffle=True)

val_loader = DataLoader(val_data,
                        batch_size=batch,
                        shuffle=True)

for epoch in range(start_epoch, start_epoch + epochs):
    
    epoch_loss = 0
    
    epoch_psnr = 0
    
    for i, data in enumerate(pbar := tqdm(train_loader,
                                          desc=f'Epoch {epoch+1}: ')):
        
        optimizer.zero_grad()
        
        ref = data[0][0].cuda()
        n = data[0][1].cuda()
        gt = data[1].cuda()
        
        output = model([ref,n])
        loss = loss_fn(output,gt)
        epoch_loss += loss.item()
        report_loss = epoch_loss / (i+1)
        
        # calc psnr metric
        epoch_psnr += psnr(gt,output).item() / len(train_loader)
        
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix_str(f'loss: {report_loss:.4f}')
        
    with torch.no_grad():
        
        epoch_val_loss = 0
        
        epoch_val_psnr = 0
        
        for data in val_loader:
            
            ref = data[0][0].cuda()
            n = data[0][1].cuda()
            gt = data[1].cuda()
            
            val_output = model([ref,n])
            val_loss = loss_fn(val_output,gt)
            
            epoch_val_loss += val_loss.item() / len(val_loader)
            
            # calc psnr metric
            epoch_val_psnr += psnr(gt,val_output).item() / len(val_loader)
            
    scheduler.step()
        
    print(
        f'Epoch {epoch+1} loss {report_loss:.4f}, val_loss {epoch_val_loss:.4f}'
        )
    
    writer.add_scalar('Loss/train',report_loss,epoch)
    writer.add_scalar('Loss/val',epoch_val_loss,epoch)
    
    writer.add_scalar('PSNR/train',epoch_psnr,epoch)
    writer.add_scalar('PSNR/val',epoch_val_psnr,epoch)
    
    if (epoch+1) % save_every == 0:
        torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   save_dir+f'epoch_{epoch+1:04d}.pth')
    
writer.close()