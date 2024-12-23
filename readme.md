# Reference-based Denoising with Dynamic Reference Acquisition

This is the implementation of reference-based denoising with dynamic reference acquisition ([paper link when published]). The package implements dynamic control of illumination power during image acquisition to generate data that can be restored with reference-based denoising, resulting in an order-of-magnitude reduction in photobleaching at no cost to spatiotemporal resolution.

The package features camera and microscope control via [pycro-manager](https://github.com/micro-manager/pycro-manager), serial commands, and [nidaqmx](https://github.com/ni/nidaqmx-python). The control of illumination power is implemented in a modular way to facilitate easy modification for any imaging system.

## Requirements 

The package requires, and was developed with, dependencies in requirements.txt. See PyTorch installation instructions for your system.

## Usage

1. Check control_functions.py to adjust serial / DAQ commands for your system
2. Run the GUI:
```bash
python gui.py
```
3. Set save directory and imaging parameters
4. Set the correct number of lasers used by your system (settings tab). Tick the laser to be used in dynamic illumination mode
5. Enter metadata if desired
6. Switch on selected lasers by clicking "set laser power"
7. Press "acquire" to begin acquisition. The data will be saved into the save directory, along with a metadata.txt file containing the imaging settings and user-input data

## After Acquisition

1. (optional) Set params in config.py to your model and saved weights directory
2. Run process.py with the appropriate arguments:
```bash
python process.py /path/to/data/dir /path/to/save/dir --ref_acq_freq 10
```
Note: The ref_acq_freq argument should match the reference acquisition frequency used during image acquisition (this can be found in metadata.txt in your data directory).
