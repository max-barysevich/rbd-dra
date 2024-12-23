import torch
import numpy as np
import tifffile
import argparse
import os
from tqdm import tqdm
from config import Config
from uformer import Uformer

def load_model(config, weights_path):
   model = Uformer(
       use_ref=True,
       img_dim=config.IMG_DIM,
       img_ch=config.IMG_CH,
       proj_dim=config.PROJ_DIM,
       proj_patch_dim=config.WINDOW_SIZE,
       attn_heads=config.ATTN_HEADS,
       attn_dim=config.ATTN_DIM,
       dropout_rate=config.DROPOUT,
       leff_filters=config.LEFF_FILTERS,
       n=config.STAGES
   ).to('cuda')

   checkpoint = torch.load(weights_path)
   model.load_state_dict(checkpoint['model'])
   model.eval()
   return model

def process_timelapse(data_dir, save_dir, ref_acq_freq=10):
   config = Config()

   run_dir = os.path.join(config.CHK_DIR, config.RESUME_FROM)
   files = os.listdir(run_dir)
   weights_path = os.path.join(run_dir, files[-1])
   denoiser = load_model(config, weights_path)

   if not os.path.exists(save_dir):
       os.makedirs(save_dir)

   for tiff_file in os.listdir(data_dir):
       if not tiff_file.endswith('.tif'):
           continue

       print(f"Processing {tiff_file}")

       with tifffile.TiffFile(os.path.join(data_dir, tiff_file)) as tif:
           denoiser_in = []

           for i in tqdm(range(len(tif.pages))):
               if i == 0:
                   continue

               page = tif.pages[i]
               imn = page.asarray()[np.newaxis,np.newaxis]

               if (i-1)%ref_acq_freq == 0:
                   ref = imn

               denoiser_in.append([ref,imn])

           denoiser_out = []

           for x in tqdm(denoiser_in):
               with torch.no_grad():
                   y = denoiser([torch.tensor(x[0],device='cuda',dtype=torch.float32),
                               torch.tensor(x[1],device='cuda',dtype=torch.float32)])
                   denoiser_out.append(y)

           to_save = [im[0,0].cpu().numpy() for im in denoiser_out]
           to_save = np.stack(to_save,axis=0)

           save_path = os.path.join(save_dir, os.path.splitext(tiff_file)[0] + '_denoised.tif')
           tifffile.imwrite(save_path, to_save)
           print(f"Saved to {save_path}")

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Reference-based denoising for timelapse data')
   parser.add_argument('data_dir', type=str, help='Directory containing .tif files to process')
   parser.add_argument('save_dir', type=str, help='Directory to save processed files')
   parser.add_argument('--ref_acq_freq', type=int, default=10, help='Reference acquisition frequency')

   args = parser.parse_args()
   process_timelapse(args.data_dir, args.save_dir, args.ref_acq_freq)
