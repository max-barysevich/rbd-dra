# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import multiprocessing as mp
import threading
import numpy as np
import control_functions as cf
from pycromanager import Core
#from dummy_hardware import Core
import time
import os
import json
import datetime

class GUI:
    def __init__(self,master):
        self.master = master
        master.title('Dynamic Reference Acquisition')
        master.geometry('1200x1000')
        tabcontrol = ttk.Notebook(master)

        self.acquisition_running = False
        self.liveview_running = False
        self.stop_signal = mp.Queue()
        self.output = mp.Queue()
        self.stack = mp.Queue()
        self.display = mp.Queue()

        self.tab1 = ttk.Frame(tabcontrol)
        tabcontrol.add(self.tab1,text='imaging')
        tabcontrol.place(x = 5,y = 5, width = 1000, height = 930)

        self.tab2 = ttk.Frame(tabcontrol)
        tabcontrol.add(self.tab2,text='metadata')
        tabcontrol.place(x=25,y=5,width=1000,height=930)

        self.tab3 = ttk.Frame(tabcontrol)
        tabcontrol.add(self.tab3,text='settings')
        tabcontrol.place(x=45,y=5,width=1000,height=930)

        self.save_dir_button = tk.Button(self.tab1,
                                         width=8,
                                         text='Browse',
                                         command = self.choose_savedir)
        self.save_dir_button.place(x=10, y=80)
        self.savedir = tk.StringVar()
        self.savedir.set('./default_save_dir/')
        self.save_dir_entry = tk.Entry(self.tab1,textvariable=self.savedir)
        self.save_dir_entry.place(x=10,y=40,width=345)
        self.save_dir_label = tk.Label(self.tab1,text = 'Save to:')
        self.save_dir_label.place(x=10,y=10)

        self.exposure_label = tk.Label(self.tab1, text = "Exposure time (ms)")
        self.exposure_label.place(x = 10,y = 150)
        self.exposure_var = tk.IntVar()
        self.exposure_var.set(100)
        self.exposure_entry = tk.Entry(self.tab1,textvariable=self.exposure_var) # exposure time field
        self.exposure_entry.place(x=320, y=150, width=35)

        self.ref_freq_label = tk.Label(self.tab1,text = 'Reference acquisition frequency')
        self.ref_freq_label.place(x=10,y=200)
        self.ref_freq_var = tk.IntVar()
        self.ref_freq_var.set(10)
        self.ref_freq_entry = tk.Entry(self.tab1,textvariable=self.ref_freq_var)
        self.ref_freq_entry.place(x=320,y=200,width=35)

        self.exp_ratio_label = tk.Label(self.tab1,text = 'Exposure time ratio')
        self.exp_ratio_label.place(x=10,y=250)
        self.exp_ratio_var = tk.IntVar()
        self.exp_ratio_var.set(1)
        self.exp_ratio_entry = tk.Entry(self.tab1,textvariable=self.exp_ratio_var)
        self.exp_ratio_entry.place(x=320,y=250,width=35)

        self.illum_ratio_label = tk.Label(self.tab1,text = 'Laser power ratio')
        self.illum_ratio_label.place(x=10,y=300)
        self.illum_ratio_var = tk.IntVar()
        self.illum_ratio_var.set(20)
        self.illum_ratio_entry = tk.Entry(self.tab1,textvariable=self.illum_ratio_var)
        self.illum_ratio_entry.place(x=320,y=300,width=35)

        # laser power percentage (live only)
        self.illum_label = tk.Label(self.tab1,
                                    text='Laser power percentage (live only)')
        self.illum_label.place(x=10,y=350)
        self.illum_var = tk.IntVar()
        self.illum_var.set(100)
        self.illum = tk.Entry(self.tab1,textvariable=self.illum_var)
        self.illum.place(x=320,y=350,width=35)

        # switching lasers selected in settings
        self.laser_switch_var = tk.BooleanVar()
        self.laser_switch_var.set(False)
        self.laser_switch_checkbox = tk.Checkbutton(self.tab1,
                                                    text='Switch lasers',
                                                    variable=self.laser_switch_var,
                                                    command=self.switch_lasers)
        self.laser_switch_checkbox.place(x=10,y=400)

        # set the power of all lasers
        self.set_laser_power_button = tk.Button(self.tab1,
                                                width=18,
                                                height=1,
                                                text='set laser power',
                                                command = self.set_laser_power)
        self.set_laser_power_button.place(x = 10, y = 450)

        self.sim_var = tk.BooleanVar()
        self.sim_var.set(False)
        self.sim_checkbox = tk.Checkbutton(self.tab1,
                                           width=18,
                                           height=1,
                                           text='Use OS-SIM',
                                           variable=self.sim_var)
        #self.sim_checkbox.place(x=170,y=500)

        self.start_button = tk.Button(self.tab1,
                                      width=8,
                                      text='Acquire',
                                      command=self.acquisition)
        self.start_button.place(x=10, y=500)

        self.live_button = tk.Button(self.tab1,
                                     width=8,
                                     text='Live',
                                     command=self.liveview)
        self.live_button.place(x=90,y=500)

        self.stop_button = tk.Button(self.tab1,
                                     width=8,
                                     text='Stop',
                                     command=self.stop_acquisition)
        self.stop_button.place(x=10, y=550)

        self.quit_button = tk.Button(self.tab1,
                                     width=8,
                                     text='Quit',
                                     command=self.quit_gui)
        self.quit_button.place(x=90, y=550)

        self.default_liveview = ImageTk.PhotoImage(
            image=Image.fromarray(np.zeros((512,512),dtype='uint8'))
            )
        self.videofeed = tk.Label(self.tab1,image=self.default_liveview)
        self.videofeed.place(x=400,y=40)

        self.max_val_label = tk.Label(self.tab1,text='Maximum value:')
        self.max_val_label.place(x=400,y=10)
        self.max_val_field = tk.Label(self.tab1,text='None')
        self.max_val_field.place(x=520,y=10)

        # metadata entry fields - sample, dye, other
        self.sample_label = tk.Label(self.tab2,text = 'Sample: ')
        self.sample_label.place(x=10,y=150)
        self.sample_var = tk.StringVar()
        self.sample_var.set(None)
        self.sample_entry = tk.Entry(self.tab2,textvariable=self.sample_var)
        self.sample_entry.place(x=120,y=150,width=200)

        self.dye_label = tk.Label(self.tab2,text = 'Staining: ')
        self.dye_label.place(x=10,y=200)
        self.dye_var = tk.StringVar()
        self.dye_var.set(None)
        self.dye_entry = tk.Entry(self.tab2,textvariable=self.dye_var)
        self.dye_entry.place(x=120,y=200,width=200)

        self.other_label = tk.Label(self.tab2,text = 'Other: ')
        self.other_label.place(x=10,y=250)
        self.other_var = tk.StringVar()
        self.other_var.set(None)
        self.other_entry = tk.Entry(self.tab2,textvariable=self.other_var)
        self.other_entry.place(x=120,y=250,width=200)

        self.pix_label = tk.Label(self.tab2,text = 'Pixel size: ')
        self.pix_label.place(x=10,y=300)
        self.pix_var = tk.StringVar()
        self.pix_var.set(None)
        self.pix_entry = tk.Entry(self.tab2,textvariable=self.pix_var)
        self.pix_entry.place(x=120,y=300,width=200)

        # Settings

        self.num_shutters_label = tk.Label(self.tab3,text='Number of shutters:')
        self.num_shutters_label.place(x=10,y=20)
        self.num_shutters_var = tk.IntVar()
        self.num_shutters_var.set(3)
        self.num_shutters_entry = tk.Entry(self.tab3,textvariable=self.num_shutters_var)
        self.num_shutters_entry.place(x=160, y=20, width=20)
        self.shutter_gen_button = tk.Button(self.tab3,
                                            width=10,
                                            text='Regenerate',
                                            command=self.create_shutter_checkboxes)
        self.shutter_gen_button.place(x=10,y=50)
        self.shutter_checkboxes = []
        self.create_shutter_checkboxes()

        self.num_lasers_label = tk.Label(self.tab3,text='Number of lasers:')
        self.num_lasers_label.place(x=320,y=20)
        self.num_lasers_var = tk.IntVar()
        self.num_lasers_var.set(2)
        self.num_lasers_entry = tk.Entry(self.tab3,textvariable=self.num_lasers_var)
        self.num_lasers_entry.place(x=470,y=20,width=20)
        self.laser_gen_button = tk.Button(self.tab3,
                                          width=10,
                                          text='Regenerate',
                                          command=self.create_laser_checkboxes)
        self.laser_gen_button.place(x=320,y=50)
        self.laser_checkboxes = []
        self.create_laser_checkboxes()

        # FOV settings

        self.ROI_label = tk.Label(self.tab3,text='ROI:')
        self.ROI_label.place(x=640,y=20)

        self.ROI_x_label = tk.Label(self.tab3,text='x:')
        self.ROI_x_label.place(x=640,y=50)
        self.ROI_x_var = tk.IntVar()
        self.ROI_x_var.set(730)
        self.ROI_x_entry = tk.Entry(self.tab3,
                                    width=6,
                                    textvariable=self.ROI_x_var)
        self.ROI_x_entry.place(x=700,y=50)

        self.ROI_y_label = tk.Label(self.tab3,text='y:')
        self.ROI_y_label.place(x=640,y=80)
        self.ROI_y_var = tk.IntVar()
        self.ROI_y_var.set(600)
        self.ROI_y_entry = tk.Entry(self.tab3,
                                    width=6,
                                    textvariable=self.ROI_y_var)
        self.ROI_y_entry.place(x=700,y=80)

        # width and height - will need to rewrite liveview

    def set_laser_power(self):
        print('Setting laser power by calling cf.set_laser_power.')
        cf.set_laser_power()
        print('Laser power set.')

    def choose_savedir(self):
        self.savedir.set(filedialog.askdirectory())
        print(f'Saving to {self.savedir.get()}.')

    def create_shutter_checkboxes(self):
        # .destroy() existing
        for var,checkbox in self.shutter_checkboxes:
            del var
            checkbox.destroy()
        self.shutter_checkboxes.clear()
        # append new
        for i in range(self.num_shutters_var.get()):
            var = tk.BooleanVar()
            var.set(False)
            checkbox = tk.Checkbutton(self.tab3,
                                      text='Shutter '+str(i+1),
                                      variable=var)
            checkbox.place(x=10,y=85+30*i)
            self.shutter_checkboxes.append((var,checkbox))

    def create_laser_checkboxes(self):
        for var,checkbox in self.laser_checkboxes:
            del var
            checkbox.destroy()
        self.laser_checkboxes.clear()

        for i in range(self.num_lasers_var.get()):
            var = tk.BooleanVar()
            var.set(False)
            checkbox = tk.Checkbutton(self.tab3,
                                      text='Laser '+str(i+1),
                                      variable=var)
            checkbox.place(x=320,y=85+30*i)
            self.laser_checkboxes.append((var,checkbox))

    def switch_lasers(self):

        switch_state = self.laser_switch_var.get()

        lasers = [var.get() for var,_ in self.shutter_checkboxes]

        cf.switch_lasers(switch_state,lasers)

    def acquisition(self):

        use_sim = self.sim_var.get()

        if use_sim:
            self.acquisition_sim()
        else:
            self.acquisition_wf()

    def acquisition_wf(self):
        print('Setting up acquisition - widefield.')
        self.acquisition_running = True

        self.disable_buttons()

        core = Core()

        x1 = self.ROI_x_var.get()
        y1 = self.ROI_y_var.get()
        width = 512
        height = 512
        ROI = [x1, y1, width, height] # build ROI
        core.set_roi(*ROI) # set ROI
        print('Successfully set ROI')

        date_time = datetime.datetime.now().strftime('saved_%d%m%YT%H%M')
        save_dir = os.path.join(self.savedir.get(),date_time)
        print(f'Saving to {save_dir}.')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        metadata = {
            'Metadata':{
                'Directory': self.savedir.get(),
                'Datetime': date_time,
                'Reference exposure time': self.exposure_var.get(),
                'Reference acquisition frequency': self.ref_freq_var.get(),
                'Exposure time ratio': self.exp_ratio_var.get(),
                'Laser power ratio': self.illum_ratio_var.get(),
                'Pixel size': None,
                'FOV coordinates': (x1,y1),
                'FOV height': height,
                'FOV width': width,
                'Sample': None,
                'Dye': None,
                'Other': None
                }
            }

        # save the metadata file
        with open(save_dir+'/metadata.txt','w') as file:
            file.write(json.dumps(metadata,indent=4))

        exposure_time = self.exposure_var.get()
        ref_freq = self.ref_freq_var.get()
        exposure_ratio = self.exp_ratio_var.get()
        illum_ratio = self.illum_ratio_var.get()
        #laser_hack = self.laser_hack_var.get()
        laser_hack = [var.get() for var,_ in self.laser_checkboxes]

        self.live_process = mp.Process(target=cf.acquire_and_save,args=(
            self.stack,
            self.stop_signal,
            self.output,
            exposure_time,
            ref_freq,
            exposure_ratio,
            illum_ratio,
            laser_hack,
            save_dir))

        self.display_process = threading.Thread(target=self.acquisition_plot)

        self.live_process.start()
        self.display_process.start()

    def acquisition_sim(self):
        print('Setting up acquisition - OS-SIM.')
        self.acquisition_running = True

        self.disable_buttons()

        core = Core()

        x1 = self.ROI_x_var.get()
        y1 = self.ROI_y_var.get()
        width = 512
        height = 512
        ROI = [x1, y1, width, height] # build ROI
        core.set_roi(*ROI) # set ROI
        print('Successfully set ROI')

        date_time = datetime.datetime.now().strftime('saved_%d%m%YT%H%M')
        save_dir = os.path.join(self.savedir.get(),date_time)
        print(f'Saving to {save_dir}.')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        metadata = {
            'Metadata':{
                'Directory': self.savedir.get(),
                'Datetime': date_time,
                'Reference exposure time': self.exposure_var.get(),
                'Reference acquisition frequency': self.ref_freq_var.get(),
                'Exposure time ratio': self.exp_ratio_var.get(),
                'Laser power ratio': self.illum_ratio_var.get(),
                'Pixel size': None,
                'FOV coordinates': (x1,y1),
                'FOV height': height,
                'FOV width': width,
                'Sample': None,
                'Dye': None,
                'Other': None
                }
            }

        # save the metadata file
        with open(save_dir+'/metadata.txt','w') as file:
            file.write(json.dumps(metadata,indent=4))

        exposure_time = self.exposure_var.get()
        ref_freq = self.ref_freq_var.get()
        exposure_ratio = self.exp_ratio_var.get()
        illum_ratio = self.illum_ratio_var.get()
        laser_hack = [var.get() for var,_ in self.laser_checkboxes]

        print('Creating main process.')

        self.live_process = mp.Process(target=cf.acquire_sim_and_save,args=(
            self.stack,
            self.stop_signal,
            self.output,
            exposure_time,
            ref_freq,
            exposure_ratio,
            illum_ratio,
            laser_hack,
            save_dir))

        print('Creating display process.')

        print('Starting main process.')
        self.live_process.start()

    def acquisition_plot(self):

        while True:
            if not self.output.empty():
                disp = self.output.get()
                # clear output queue if it grows
                if isinstance(disp,bool):
                    break

                if len(disp.shape) == 3:
                    disp = np.max(disp,axis=0)

                disp_max = np.max(disp)
                q = np.percentile(disp,99)
                disp = (255 * np.clip(np.rot90(disp,k=1),0,q)/q).astype('uint8')

                disp_img = ImageTk.PhotoImage(image=Image.fromarray(disp))

                # update image
                self.videofeed.configure(image=disp_img)
                # update max value label
                self.max_val_field.configure(text=str(disp_max))

    def liveview(self):
        print('Setting up liveview.')
        self.liveview_running = True

        self.disable_buttons()

        core = Core()

        x1 = self.ROI_x_var.get()
        y1 = self.ROI_y_var.get()
        width = 512
        height = 512
        ROI = [x1, y1, width, height] # build ROI
        core.set_roi(*ROI) # set ROI
        print('Successfully set ROI')

        exposure = self.exposure_var.get()
        illum = self.illum_var.get()
        #laser_hack = self.laser_hack_var.get()
        laser_hack = [var.get() for var,_ in self.laser_checkboxes]

        self.liveview_process = mp.Process(target=cf.live_loop,
                                           args=(self.stop_signal,
                                                 self.display,
                                                 exposure,
                                                 illum,
                                                 laser_hack))

        self.display_process = threading.Thread(target=self.liveview_plot)

        self.liveview_process.start()
        self.display_process.start()

    def liveview_plot(self):

        while True:
            if not self.display.empty():
                disp = self.display.get()
                if isinstance(disp,bool):
                    break

                disp_max = np.max(disp)
                # normalise before displaying
                q = np.percentile(disp,99)
                disp = (255 * np.clip(np.rot90(disp,k=1),0,q)/q).astype('uint8')

                disp_img = ImageTk.PhotoImage(image=Image.fromarray(disp))

                # update image
                self.videofeed.configure(image=disp_img)
                # update max value label
                self.max_val_field.configure(text=str(disp_max)) # or q

        self.liveview_running = False

    def disable_buttons(self):
        self.start_button['state'] = tk.DISABLED
        self.live_button['state'] = tk.DISABLED
        self.quit_button['state'] = tk.DISABLED

    def stop_acquisition(self):
        self.stop_signal.put(False)

        lasers = [True for var,_ in self.shutter_checkboxes]

        cf.switch_lasers(False,lasers)

        if self.acquisition_running:
            print('Waiting for the acquisition processes to terminate.')
            while not isinstance(self.output.get(),bool):
                time.sleep(.5)
            self.acquisition_running = False
        '''
        elif self.liveview_running:
            print('Waiting for the liveview processes to terminate.')
            while self.liveview_running:
                time.sleep(.5)
        '''
        self.start_button['state'] = tk.NORMAL
        self.live_button['state'] = tk.NORMAL
        self.quit_button['state'] = tk.NORMAL

    def quit_gui(self):

        print('Quitting.')

        self.master.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    my_gui = GUI(root)
    root.mainloop()
