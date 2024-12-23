import numpy as np

class Core:

    def __init__(self):
        self.sequence_is_running = False
        self._image_count_calls = 0

    def get_remaining_image_count(self):
        if self._image_count_calls == 0:
            self._image_count_calls += 1
            return 0
        else:
            self._image_count_calls = 0
            return 1

    def is_sequence_running(self):
        return self.sequence_is_running

    def initialize_circular_buffer(self):
        pass

    def set_exposure(self, exposure):
        pass

    def start_continuous_sequence_acquisition(self,delay):
        self.sequence_is_running = True

    def pop_next_tagged_image(self):
        return taggedImage()

    def load_system_configuration(self,config):
        pass

    def get_property(self,*args):
        pass

    def set_property(self,*args):
        pass

    def set_roi(self,*args):
        pass

    def get_last_tagged_image(self):
        return taggedImage()

    def get_remaining_image_count(self):
        return 1

    def stop_sequence_acquisition(self):
        self.sequence_is_running = False

class taggedImage:

    def __init__(self):
        self.pix = np.random.randint(0,2**16,(512*512,)).astype(np.uint16)
        self.tags = {'Height':512,'Width':512}

class Task:

    def __init__(self):
        self.do_channels = DAQ_channel()
        self.ao_channels = DAQ_channel()

    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        self.close()

    def write(self, data):
        pass

    def close(self):
        pass

class DAQ_channel:

    def __init__(self):
        pass

    def add_do_chan(self,chan):
        pass

    def add_ao_voltage_chan(self,chan):
        pass

class DAQ:
    Task = Task

class Serial:
    def __init__(self, port, baudrate, bytesize, parity, stopbits, timeout, xonxoff, rtscts, dsrdtr, write_timeout):
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def write(self, data):
        pass

    def readline(self):
        return b'OK\r\n'  # Simulate successful response

    def close(self):
        pass
