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
        self.csv_files = []

    def extract_order(self, filename):
        filename = os.path.basename(filename)
        match = re.search(fr'(\d+)-{self.label}\.{self.ext}$', filename)
        if match:
            return float(match.group(1))
        return 0

    def order_sort(self, names):
        return sorted(names, key=lambda filename: (self.extract_order(filename), filename))

    def _resolve_memmap_path(self):
        memmap_path = os.path.expanduser(self.memmap_filename)
        if os.path.isabs(memmap_path):
            return memmap_path

        cache_dir = os.path.join(os.path.abspath(self.data_dir), ".alibz_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, os.path.basename(memmap_path))

    def _load_file(self, file_path, w_col=0, i_col=1):
        if w_col == i_col:
            raise ValueError("w_col and i_col must refer to different columns")
        if min(w_col, i_col) < 0:
            raise ValueError("w_col and i_col must be non-negative")

        data = np.loadtxt(
            file_path,
            delimiter=self.delimiter,
            skiprows=1,
            dtype=self.dtype,
        )
        data = np.asarray(data, dtype=self.dtype)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D tabular data in {file_path!r}")
        if data.shape[1] <= max(w_col, i_col):
            raise ValueError(
                f"File {file_path!r} does not contain requested columns "
                f"{w_col} and {i_col}"
            )

        return data[:, [w_col, i_col]].T

    def load_data(self, w_col=0, i_col=1):
        lib_path = self.data_dir
        csv_files = []
        for root, dirs, files in os.walk(lib_path):
            for file in files:
                if file.endswith('.' + self.ext):
                    csv_files.append(os.path.join(root, file))

        sorted_csv_files = self.order_sort(csv_files)
        if len(sorted_csv_files) == 0:
            raise FileNotFoundError(
                f"No .{self.ext} files found under {os.path.abspath(lib_path)!r}"
            )

        self.csv_files = sorted_csv_files
        num_files = len(sorted_csv_files)
        sample_data = self._load_file(sorted_csv_files[0], w_col=w_col, i_col=i_col)
        self.shape = (num_files, sample_data.shape[0], sample_data.shape[1])

        if self.memmap:
            self.data = np.memmap(
                self._resolve_memmap_path(),
                dtype=self.dtype,
                mode='w+',
                shape=self.shape,
            )
        else:
            self.data = np.zeros(self.shape, dtype=self.dtype)

        for i, file_path in enumerate(sorted_csv_files):
            data = self._load_file(file_path, w_col=w_col, i_col=i_col)
            if data.shape != sample_data.shape:
                raise ValueError(
                    f"Inconsistent spectrum shape for {file_path!r}: "
                    f"expected {sample_data.shape}, got {data.shape}"
                )
            self.data[i, :, :] = data

        return self.data

    def get_data(self):
        if self.data is None:
            return self.load_data()
        return self.data

    def close_memmap(self):
        if self.memmap and self.data is not None:
            self.data.flush()
            self.data._mmap.close()
            self.data = None
