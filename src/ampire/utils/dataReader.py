from sklearn import datasets
import pandas as pd
import numpy as np
import scipy.io

class DataReader:
    def __init__(self, file, make=False):
        self.file = file
        self.make_ = make
        self.format = file.split('.')[-1] if '.' in file else 'sklearn'

    def read_mat(self, *args, **kwargs):
        return scipy.io.loadmat(self.file, *args, **kwargs)

    def read_csv(self, skiprows=None, *args, **kwargs):
        data = pd.read_csv(self.file, skiprows=skiprows, *args, **kwargs)
        if skiprows is None:
            return data
        return np.array(data)

    def make(self, *args, **kwargs):
        method_name = 'make_' + self.file
        return getattr(datasets, method_name)(*args, **kwargs)

    def load(self, dataset: str = 'iris', *args, **kwargs):
        method = f'load_{dataset}'
        return getattr(datasets, method)(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.name = 'make' if self.make_ else 'read_' + self.format

        if hasattr(self, self.name) and getattr(self, self.name):
            return getattr(self, self.name)(*args, **kwargs)
        else:
            # Fallback for sklearn datasets
            return self.load(self.file, *args, **kwargs)