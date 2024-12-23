# -*- coding: utf-8 -*-

import nidaqmx
#from dummy_hardware import DAQ as nidaqmx
import numpy as np
import tifffile
import time
import datetime
import os
import multiprocessing as mp
from pycromanager import Core
#from dummy_hardware import Core
import serial
#from dummy_hardware import Serial as serial

mp.freeze_support()

def switch_shutter(open_shutter,
                   chan='Dev1/port0/line1'):
    with nidaqmx.Task() as ShutterTask:
        ShutterTask.do_channels.add_do_chan(chan) # line 1 for 488, line 0 for 561;
        if open_shutter:
            ShutterTask.write(True)
        else:
            ShutterTask.write(False)

def switch_647_laser(on):
    with serial.Serial(port='COM5',
                       baudrate=115200,
                       bytesize=serial.EIGHTBITS,
                       parity=serial.PARITY_NONE,
                       stopbits=serial.STOPBITS_ONE,
                       timeout=5,
                       xonxoff=False,
                       rtscts=False,
                       dsrdtr=False,
                       write_timeout=0) as ser:
        if on:

            ser.write(str.encode('en 1\r\n'))
            print(ser.readline().decode('ascii'))

            ser.write(str.encode('la on\r\n'))
            print(ser.readline().decode('ascii'))
        else:
            ser.write(str.encode('la off\r\n'))
            print(ser.readline().decode('ascii'))

def switch_lasers(on,lasers):
    '''
    Simultaneously switches shutters of selected lasers.
    Argument: lasers - List (bool) - which lasers to control.
    Rewrite for your system.
    '''

    if lasers[0]: # 488
        with nidaqmx.Task() as ShutterTask:
            ShutterTask.do_channels.add_do_chan('Dev1/port0/line1')
            ShutterTask.write(on)

    if lasers[1]: # 561
        with nidaqmx.Task() as ShutterTask:
            ShutterTask.do_channels.add_do_chan('Dev1/port0/line0')
            ShutterTask.write(on)

    if lasers[2]: # 647

        try:
            with serial.Serial(port='COM5',
                               baudrate=115200,
                               bytesize=serial.EIGHTBITS,
                               parity=serial.PARITY_NONE,
                               stopbits=serial.STOPBITS_ONE,
                               timeout=5,
                               xonxoff=False,
                               rtscts=False,
                               dsrdtr=False,
                               write_timeout=0) as ser:
                if on:

                    ser.write(str.encode('en 1\r\n'))
                    print(ser.readline().decode('ascii'))

                    ser.write(str.encode('la on\r\n'))
                    print(ser.readline().decode('ascii'))
                else:
                    # might need 'en 1\r\n' here as well
                    ser.write(str.encode('la off\r\n'))
                    print(ser.readline().decode('ascii'))

        except:
            print('Error switching the 647 nm laser.')
            # return actual error

def set_laser_power():
    '''
    Sets the maximum power of all lasers available to the system. Rewrite this
    according to how laser control is implemented on your system.
    '''
    print('Setting 488 nm laser power.')
    try:
        with serial.Serial(port='COM7',
                           baudrate=115200,
                           bytesize=serial.EIGHTBITS,
                           parity=serial.PARITY_NONE,
                           stopbits=serial.STOPBITS_ONE,
                           timeout=5,
                           xonxoff=False,
                           rtscts=False,
                           dsrdtr=False,
                           write_timeout=0) as ser:

            ser.write(str.encode('p 0.1\r'))
            print(ser.readline().decode('ascii'))
            time.sleep(1)
            ser.write(str.encode('pa?\r'))
            print(ser.readline().decode('ascii'))
    except:
        print('Error setting 488 nm laser power.')

    print('Setting 647 nm laser power.')
    try:
        with serial.Serial(port='COM5',
                           baudrate=115200,
                           bytesize=serial.EIGHTBITS,
                           parity=serial.PARITY_NONE,
                           stopbits=serial.STOPBITS_ONE,
                           timeout=5,
                           xonxoff=False,
                           rtscts=False,
                           dsrdtr=False,
                           write_timeout=0) as ser:


            ser.write(str.encode('ch 1 pow 100\r\n'))
            print(ser.readline().decode('ascii'))
    except:
        print('Error setting 647 nm laser power.')

class LaserTaskHack(object):
    '''
    Enables simultaneous control of multiple lasers via DAQ and RS-232.
    Argument: lasers - Tuple (bool) - which laser(s) to control.
    Rewrite for your system.
    '''

    def __init__(self,lasers):

        self.lasers = lasers

        if lasers[0]:
            self.laser1 = nidaqmx.Task()
            self.laser1.ao_channels.add_ao_voltage_chan('Dev1/ao1')

        if lasers[1]:
            self.laser2 = serial.Serial(port='COM5',
                                        baudrate=115200,
                                        bytesize=serial.EIGHTBITS,
                                        parity=serial.PARITY_NONE,
                                        stopbits=serial.STOPBITS_ONE,
                                        timeout=5,
                                        xonxoff=False,
                                        rtscts=False,
                                        dsrdtr=False,
                                        write_timeout=0)

    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        self.close()

    def write(self,signal):

        if self.lasers[0]:
            self.laser1.write(signal)

        if self.lasers[1]:
            target_power = signal * 100 / 9
            self.laser2.write(str.encode('ch 1 pow '+str(target_power)+'\r\n'))

    def close(self):

        if self.lasers[0]:
            self.laser1.close()

        if self.lasers[1]:
            self.laser2.close()

def save_loop(stack,output,save_dir):

    with tifffile.TiffWriter(save_dir+'/'+'data.tif') as tif:
        while True:
            if not output.empty():
                _ = output.get()
                # needed for preventing output from growing
                # instead send to a new display loop
            if not stack.empty():
                pixels = stack.get()
                if isinstance(pixels,bool):
                    break
                else:
                    output.put(pixels)
                    tif.write(pixels.astype(np.int16),contiguous=True)
                    time.sleep(0.001)
    output.put(False)

def acquisition_loop(stop_signal,stack,exposure,
                     ref_freq,exposure_ratio,illum_ratio,
                     laser_hack):
    print('Starting acquisition - widefield.')
    stop_signal.put(True)

    with (nidaqmx.Task() as VoltageTask,
          nidaqmx.Task() as CameraTask,
          LaserTaskHack(laser_hack) as LaserTask):
        voltages = 2.2
        waits = 0.001
        laser_signal = 9. # max 10 V

        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")

        core = Core()

        if core.is_sequence_running():

            CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
            time.sleep(0.5/1000)
            CameraTask.write(False)
            core.stop_sequence_acquisition() # stop the camera

        core.initialize_circular_buffer()

        core.start_continuous_sequence_acquisition(0) # start the camera

        CameraTask.write(True) # tell camera to take image
        time.sleep(exposure/1000)
        CameraTask.write(False)
        while core.get_remaining_image_count() == 0: #wait until picture is available
            time.sleep(0.001)
        result = core.get_last_tagged_image() # get image data into python

        VoltageTask.write(voltages) # move galvo
        time.sleep(waits)

        while True:

            status = stop_signal.get()
            if status == False:
                break
            else:
                stop_signal.put(True)
                if stack.empty():
                    for i in range(ref_freq):
                        CameraTask.write(True)
                        if i == 0:
                            LaserTask.write(laser_signal)
                            time.sleep(exposure/1000)
                            CameraTask.write(False)
                        else:
                            LaserTask.write(laser_signal/illum_ratio)
                            time.sleep(exposure/(1000*exposure_ratio))
                            CameraTask.write(False)
                        while core.get_remaining_image_count() == 0: # wait until image is available
                            time.sleep(0.001)
                        result = core.pop_next_tagged_image() # get image data into python
                        pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data

                        if i == 0:
                            merged = np.zeros([ref_freq,pixels.shape[0],pixels.shape[1]])
                            merged[0,:,:] = pixels.copy()
                        else:
                            merged[i,:,:] = pixels.copy()

                        if stack.empty():
                            stack.put(pixels)

                    #to_rest.put(pixels) # to restoration loop
        print('Stopping acquisition.')

        CameraTask.write(True) # make sure camera has stopped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        core.stop_sequence_acquisition() # stop the camera

        print('Camera stopped.')

        stack.put(False)

def acquisition_loop_sim(stop_signal,stack,exposure,
                     ref_freq,exposure_ratio,illum_ratio,
                     laser_hack):
    print('Starting acquisition - OS SIM.')
    stop_signal.put(True)

    with (nidaqmx.Task() as VoltageTask,
          nidaqmx.Task() as CameraTask,
          LaserTaskHack(laser_hack) as LaserTask):

        voltages = [2.2002, 2.2018, 2.2036]
        waits = [.001, .001, .001]
        laser_signal = 9. # max 10 V

        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")

        print('Starting camera.')

        core = Core()

        if core.is_sequence_running():

            CameraTask.write(True) # make sure camera has stoppped by requesting a final unused image
            time.sleep(0.5/1000)
            CameraTask.write(False)
            core.stop_sequence_acquisition() # stop the camera

        core.initialize_circular_buffer()

        core.start_continuous_sequence_acquisition(0) # start the camera

        CameraTask.write(True) # tell camera to take image
        time.sleep(exposure/1000)
        CameraTask.write(False)
        while core.get_remaining_image_count() == 0: #wait until picture is available
            time.sleep(0.001)
        result = core.get_last_tagged_image() # get image data into python

        print('Starting acquisition loop.')

        while True:

            status = stop_signal.get()

            if not status:
                break
            else:

                stop_signal.put(True)

                if stack.empty():

                    for i in range(ref_freq):

                        # make sim stack here
                        sim_frame = np.zeros((3,result.tags['Height'],result.tags['Width']))

                        for j in range(3):

                            VoltageTask.write(voltages[j])
                            time.sleep(waits[j])

                            CameraTask.write(True)

                            if i == 0:
                                LaserTask.write(laser_signal)
                                time.sleep(exposure/1000)
                                CameraTask.write(False)
                            else:
                                LaserTask.write(laser_signal/illum_ratio)
                                time.sleep(exposure/(1000*exposure_ratio))
                                CameraTask.write(False)

                            while core.get_remaining_image_count() == 0: # wait until image is available
                                time.sleep(0.001)
                            result = core.pop_next_tagged_image() # get image data into python
                            pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data

                            # update sim stack here
                            sim_frame[j,:,:] = pixels.copy()

                        if stack.empty():
                            stack.put(sim_frame)

        print('Stopping acquisition.')

        CameraTask.write(True) # make sure camera has stopped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        core.stop_sequence_acquisition() # stop the camera

        print('Camera stopped.')

        stack.put(False)

def live_loop(stop_signal,display,exposure,illum,
              laser_hack):
    print('Starting liveview.')
    stop_signal.put(True)

    with (nidaqmx.Task() as VoltageTask,
          nidaqmx.Task() as CameraTask,
          LaserTaskHack(laser_hack) as LaserTask):
        voltages = 2.2
        waits = 0.001
        laser_signal = 9. # max 10 V
        #illum = np.clip(illum,0,100)

        VoltageTask.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        CameraTask.do_channels.add_do_chan("Dev1/port0/line2")
        #if not laser_hack:
        #    LaserTask.ao_channels.add_ao_voltage_chan("Dev1/ao1")

        VoltageTask.write(voltages) # move microscope
        time.sleep(waits)
        LaserTask.write(laser_signal*illum/100)

        core = Core()

        if core.is_sequence_running():

            CameraTask.write(True) # make sure camera has stopped by requesting a final unused image
            time.sleep(0.5/1000)
            CameraTask.write(False)
            core.stop_sequence_acquisition() # stop the camera

        core.initialize_circular_buffer()

        core.start_continuous_sequence_acquisition(0) # start the camera

        CameraTask.write(True) # tell camera to take image
        time.sleep(exposure/1000)
        CameraTask.write(False)
        while core.get_remaining_image_count() == 0: #wait until picture is available
            time.sleep(0.001)
        result = core.get_last_tagged_image() # get image data into python

        while True:

            status = stop_signal.get()
            if status == False:
                break
            else:
                stop_signal.put(True)
                if display.empty():
                    # only acquire images as fast as they can be displayed
                    CameraTask.write(True)
                    time.sleep(exposure/1000)
                    CameraTask.write(False)

                    while core.get_remaining_image_count() == 0: # wait until image is available
                        time.sleep(0.001)

                    result = core.pop_next_tagged_image() # get image data into python
                    pixels = np.squeeze(np.reshape(result.pix,newshape=[-1, result.tags["Height"], result.tags["Width"]],)) # reshape image data

                    display.put(pixels)

        print('Stopping acquisition.')

        CameraTask.write(True) # make sure camera has stopped by requesting a final unused image
        time.sleep(0.5/1000)
        CameraTask.write(False)
        core.stop_sequence_acquisition() # stop the camera

        print('Camera stopped.')

        display.put(False)


def acquire_and_save(stack,
                     stop_signal,
                     output,
                     exposure,
                     ref_freq,
                     exposure_ratio,
                     illum_ratio,
                     laser_hack,
                     savedir):
    print('Starting acquisition (widefield) and saving processes.')
    processes = [] # initialise processes
    proc_live = mp.Process(target=acquisition_loop, args=(stop_signal,
                                                          stack,
                                                          exposure,
                                                          ref_freq,
                                                          exposure_ratio,
                                                          illum_ratio,
                                                          laser_hack))
    processes.append(proc_live)
    proc_save = mp.Process(target=save_loop, args=(stack,output,savedir))
    processes.append(proc_save)
    processes.reverse()

    for process in processes:
        process.start()
    for process in processes:
        process.join()

def acquire_sim_and_save(stack,
                         stop_signal,
                         output,
                         exposure,
                         ref_freq,
                         exposure_ratio,
                         illum_ratio,
                         laser_hack,
                         savedir):
    print('Starting acquisition (OS-SIM) and saving processes.')
    processes = [] # initialise processes
    proc_live = mp.Process(target=acquisition_loop_sim, args=(stop_signal,
                                                              stack,
                                                              exposure,
                                                              ref_freq,
                                                              exposure_ratio,
                                                              illum_ratio,
                                                              laser_hack))
    processes.append(proc_live)
    proc_save = mp.Process(target=save_loop, args=(stack,output,savedir))
    processes.append(proc_save)
    processes.reverse()

    for process in processes:
        process.start()
    for process in processes:
        process.join()

def display_liveview(stop_signal,
                     display,
                     exposure,
                     illum,
                     laser_hack):
    print('Starting liveview process.')

    process = mp.Process(target=live_loop, args=(stop_signal,
                                                 display,
                                                 exposure,
                                                 illum,
                                                 laser_hack))
    process.start()
    process.join()
