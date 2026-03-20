import os
import re
import numpy as np

class Data:
    """ 'Data' is a dataloader class
        Files are optionally parsed and sorted, then loaded
        Load can be either into RAM (default) or memory mapped (memmap=True) for large datasets
        
        Args:
            data_dir (str): data directory path from which to load all data contained within
            memmap (bool): whether to use memory mapping. Default is False
            memmap_filename (str): name of memory map file. Default is 'memmap.dat'
            dtype (type): data type. Default is 'float'
        Attributes:
            data (numpy array): 
        Methods:
            extract_order
            order_sort
            load_data
            get_data
            close_memmap
    """

    def __init__(self, data_dir, label='Test', ext='csv', delimiter=',', memmap=False, memmap_filename='memmap.dat', dtype=float):
        self.data_dir = data_dir
        self.label = label
        self.ext = ext
        self.delimiter = delimiter
        self.memmap = memmap
        self.memmap_filename = memmap_filename
        self.dtype = dtype
        self.data = None
        self.shape = None

    def extract_order(self, filename):
        match = re.search(fr'(\d+)-{self.label}\.{self.ext}$', filename)
        if match:
            return float(match.group(1))
        return 0

    def order_sort(self, names):
        return sorted(names, key=lambda filename: self.extract_order(filename))

    def load_data(self, w_col=0, i_col=1):
        lib_path = self.data_dir
        csv_files = []
        for root, dirs, files in os.walk(lib_path):
            for file in files:
                if file.endswith('.' + self.ext):
                    csv_files.append(os.path.join(root, file))

        sorted_csv_files = self.order_sort(csv_files)
        num_files = len(sorted_csv_files)
        
        if self.shape is None:
            sample_data = np.loadtxt(sorted_csv_files[0], delimiter=self.delimiter, skiprows=1, dtype=self.dtype)
            n, m = sample_data.shape # features, channels
            self.shape = (num_files, m, n)
        
        if self.memmap:
            self.data = np.memmap(self.memmap_filename, dtype=self.dtype, mode='w+', shape=self.shape)
        else:
            self.data = np.zeros(self.shape, dtype=self.dtype)
        
        for i, file_path in enumerate(sorted_csv_files):
            data = np.loadtxt(file_path, delimiter=self.delimiter, skiprows=1, dtype=self.dtype)
            self.data[i, :, :] = np.swapaxes(data, 0, 1) # samples, channels, features

    def get_data(self):
        return self.data

    def close_memmap(self):
        if self.memmap and self.data is not None:
            self.data._mmap.close()