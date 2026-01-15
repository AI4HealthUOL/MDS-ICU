__all__ = ['TimeSeriesTransform', 'Compose', 'RandomCrop', 'CenterCrop', 'FixedCrop', 'GaussianNoise', 'Resample', 'ToSpectrogram', 'Flatten', 'ToTensor', 'Normalize', 'NormalizeBatch', 'ButterFilter', 'ChannelFilter', 'Transform', 'StaticTransform', 'TupleTransform', 'SequenceToSampleLabelTransform']


import numpy as np
import torch
import torch.utils.data

import math
import random
import resampy

#from skimage import transform

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy import signal

#from scipy.interpolate import interp1d



#def nn_upsample(xin, yin, xout):
#    '''performs nearest neighbor upsampling of the integer array yin with values at xin for new datapoints at xout'''
#    f = interp1d(xin,yin, kind="nearest",bounds_error=False,fill_value="extrapolate")
#    return f(xout).astype(np.int64)

#def resample_labels(startpts, labels, startpts_to_mid, startpts_new, startpts_to_mid_new):
#    '''resamples integer labels labels at starpts+startpts_to_mid to new anchor points at startpts_new+startpts_to_mid_new'''
#    if(isinstance(startpts_to_mid,float) or isinstance(startpts_to_mid,int)):
#        startpts_to_mid = np.ones_like(startpts)*startpts_to_mid
#    if(isinstance(startpts_to_mid_new,float) or isinstance(startpts_to_mid_new,int)):
#        startpts_to_mid_new = np.ones_like(startpts_new)*startpts_to_mid_new
#    midpts = np.array(startpts)+startpts_to_mid
#    midpts_new = np.array(startpts_new)+startpts_to_mid_new
#    return nn_upsample(midpts, labels, midpts_new)

#https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_filter(lowcut=10, highcut=20, fs=50, order=5, btype='band'):
    '''returns butterworth filter with given specifications'''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high] if btype=="band" else (low if btype=="low" else high), analog=False, btype=btype, output='sos')
    return sos

#def butter_filter_frequency_response(filter):
#    '''returns frequency response of a given filter (result of call of butter_filter)'''
#    w, h = sosfreqz(filter)
#    #gain vs. freq(Hz)
#    #plt.plot((fs * 0.5 / np.pi) * w, abs(h))
#    return w,h
#
#def apply_butter_filter(data, filter, forwardbackward=True):
#    '''pass filter from call of butter_filter to data (assuming time axis at dimension 0)'''
#    if(forwardbackward):
#        return sosfiltfilt(filter, data, axis=0)
#    else:
#        data = sosfilt(filter, data, axis=0)

class TimeSeriesTransform(object):
    """Base class for all time series transforms."""
    def __init__(self):
        pass

    def __call__(self, sample):
        raise NotImplementedError("__call__ must be implemented in subclasses.")

    def get_output_shape(self, input_shape, label_shape):
        """Return (output_signal_shape, output_label_shape) after transformation. Should be overridden by subclasses."""
        return input_shape, label_shape

class Compose(TimeSeriesTransform):
    '''composes several transformations into a single one (as provided by torchvision.transforms.Compose)'''
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def __call__(self, inp):
        for t in self.transforms:
            inp = t(inp)
        return inp

    def get_output_shape(self, input_shape, label_shape):
        shape = input_shape
        lshape = label_shape
        for t in self.transforms:
            if hasattr(t, 'get_output_shape'):
                shape, lshape = t.get_output_shape(shape, lshape)
        return shape, lshape


class RandomCrop(TimeSeriesTransform):
    """Crop randomly from a sample."""
    def __init__(self, output_size, annotation=False):
        super().__init__()
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        timesteps = len(data)
        assert timesteps >= self.output_size
        if timesteps == self.output_size:
            start = 0
        else:
            start = random.randint(0, timesteps - self.output_size - 1)
            idxs[1] += start
            idxs[2] = idxs[1] + self.output_size
        data = data[start: start + self.output_size]
        if self.annotation:
            label = label[start: start + self.output_size]
        return (data, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        out_signal = (self.output_size,) + input_shape[1:] if input_shape is not None else None
        if self.annotation and label_shape is not None:
            out_label = (self.output_size,) + label_shape[1:]
        else:
            out_label = label_shape
        return out_signal, out_label


class CenterCrop(TimeSeriesTransform):
    """Center crop from a sample."""
    def __init__(self, output_size, annotation=False):
        super().__init__()
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        timesteps = len(data)
        start = (timesteps - self.output_size) // 2
        idxs[1] += start
        idxs[2] = idxs[1] + self.output_size
        data = data[start: start + self.output_size]
        if self.annotation:
            label = label[start: start + self.output_size]
        return (data, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        out_signal = (self.output_size,) + input_shape[1:] if input_shape is not None else None
        if self.annotation and label_shape is not None:
            out_label = (self.output_size,) + label_shape[1:]
        else:
            out_label = label_shape
        return out_signal, out_label

class FixedCrop(TimeSeriesTransform):
    """Take a fixed crop from a sample (for aligned data e.g. spectrograms). start_idx and end_idx are relative to the start of the respective sample"""
    def __init__(self, start_idx, end_idx, annotation=False):
        super().__init__()
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.output_size = self.end_idx - self.start_idx
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        assert self.end_idx < len(data)
        start = self.start_idx
        idxs[1] += self.start_idx
        idxs[2] = idxs[1] + self.output_size
        data = data[start: start + self.output_size]
        if self.annotation:
            label = label[start: start + self.output_size]
        return (data, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        out_signal = (self.output_size,) + input_shape[1:] if input_shape is not None else None
        if self.annotation and label_shape is not None:
            out_label = (self.output_size,) + label_shape[1:]
        else:
            out_label = label_shape
        return out_signal, out_label

class GaussianNoise(TimeSeriesTransform):
    """Add gaussian noise to sample."""
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def __call__(self, sample):
        if self.scale == 0:
            return sample
        else:
            data, label, static, static_cat, idxs = sample
            data = data + np.reshape(np.array([random.gauss(0, self.scale) for _ in range(np.prod(data.shape))]), data.shape)
            return (data, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return input_shape, label_shape




class Resample(TimeSeriesTransform):
    """Resample on the fly. If annotation=True, resample integer-valued labels using nearest neighbor downsampling."""
    def __init__(self, fs_source, fs_target, annotation=False):
        super().__init__()
        self.fs_source = fs_source
        self.fs_target = fs_target
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        data = resampy.resample(data, self.fs_source, self.fs_target, axis=0)
        if self.annotation:
            # Nearest neighbor downsampling for integer-valued labels
            old_len = label.shape[0]
            new_len = int(round(old_len * float(self.fs_target) / self.fs_source))
            # Map new indices to nearest old indices
            idxs_new = np.round(np.linspace(0, old_len - 1, new_len)).astype(int)
            label = label[idxs_new]
        return (data, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        if input_shape is not None:
            new_len = int(round(input_shape[0] * float(self.fs_target) / self.fs_source))
            out_signal = (new_len,) + input_shape[1:]
        else:
            out_signal = None
        if self.annotation and label_shape is not None:
            new_label_len = int(round(label_shape[0] * float(self.fs_target) / self.fs_source))
            out_label = (new_label_len,) + label_shape[1:]
        else:
            out_label = label_shape
        return out_signal, out_label

class ToSpectrogram(TimeSeriesTransform):
    """Convert to spectrogram on the fly"""
    def __init__(self, fs, window="hann", nperseg=256, noverlap=128, pad="edge", log_transform=True):
        super().__init__()
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.pad = pad
        self.log_transform = log_transform
        self.eps = 1e-10  # More explicit than np.finfo(float).eps
        self.window = signal.get_window(window, nperseg)  # Precompute window

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        
        if self.pad is not None:
            pad_length = self.nperseg // 2
            data = np.pad(data, ((pad_length, pad_length), (0, 0)), mode=self.pad)
        
        # Calculate spectrogram: output is (n_channels, n_freqs, n_times)
        _, _, Sxx = signal.spectrogram(data.T, fs=self.fs, window=self.window,
                                        nperseg=self.nperseg, noverlap=self.noverlap)
        
        # Transpose to (n_times, n_channels, n_freqs)
        Sxx = np.transpose(Sxx, (2, 0, 1))
        
        if self.log_transform:
            Sxx = 10 * np.log10(np.maximum(Sxx, self.eps))  # 10 for power spectrum
        
        return (Sxx, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        if input_shape is not None:
            ts = input_shape[0]
            ch = input_shape[1] if len(input_shape) > 1 else 1
            pad_length = self.nperseg // 2 if self.pad is not None else 0
            ts_padded = ts + 2 * pad_length
            n_freq = self.nperseg // 2 + 1
            n_windows = 1 + (ts_padded - self.nperseg) // (self.nperseg - self.noverlap)
            out_signal = (n_windows, ch, n_freq)
        else:
            out_signal = None
        return out_signal, label_shape

class Flatten(TimeSeriesTransform):
    """flatten all channel dimensions e.g. to process spectrograms with 1d convs"""
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        #data has shape ts, ch1, ch2,...
        data = data.reshape(data.shape[0], -1)
        return (data, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        if input_shape is not None:
            out_signal = (input_shape[0], int(np.prod(input_shape[1:])))
        else:
            out_signal = None
        return out_signal, label_shape

class ToTensor(TimeSeriesTransform):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, transpose_data=True, transpose_label=False):
        super().__init__()
        #swap channel and time axis for direct application of pytorch's convs
        self.transpose_data = transpose_data
        self.transpose_label = transpose_label

    def __call__(self, sample):

        def _to_tensor(data,transpose=False):
            if(isinstance(data,np.ndarray) or isinstance(data,list)):
                if(transpose):#seq,[x,y,]ch -> [x,y,]ch, seq
                    return torch.from_numpy(np.moveaxis(data,0,-1))
                else:
                    return torch.from_numpy(data)
            else:#default_collate will take care of it
                return data

        data, label, static, static_cat, idxs = sample
        if not isinstance(data, tuple):
            data = _to_tensor(data, self.transpose_data)
        else:
            data = tuple(_to_tensor(x, self.transpose_data) for x in data)
        if not isinstance(label, tuple):
            label = _to_tensor(label, self.transpose_label)
        else:
            label = tuple(_to_tensor(x, self.transpose_label) for x in label)
        if not isinstance(static, tuple):
            static = _to_tensor(static)
        else:
            static = tuple(_to_tensor(x) for x in static)
        if not isinstance(static_cat, tuple):
            static_cat = _to_tensor(static_cat)
        else:
            static_cat = tuple(_to_tensor(x) for x in static_cat)
        if not isinstance(idxs, tuple):
            idxs = _to_tensor(idxs)
        else:
            idxs = tuple(_to_tensor(x) for x in idxs)
        return (data, label, static, static_cat, idxs) #returning as a tuple (potentially of lists)

    def get_output_shape(self, input_shape, label_shape):
        if input_shape is not None and self.transpose_data:
            out_signal = input_shape[1:] + (input_shape[0],)
        else:
            out_signal = input_shape
        if label_shape is not None and self.transpose_label:
            out_label = label_shape[1:] + (label_shape[0],)
        else:
            out_label = label_shape
        return out_signal, out_label

class Normalize(TimeSeriesTransform):
    """Normalize using given stats."""
    def __init__(self, stats_mean, stats_std, input=True, channels=[]):
        super().__init__()
        self.stats_mean = stats_mean.astype(np.float32) if stats_mean is not None else None
        self.stats_std = stats_std.astype(np.float32) + 1e-8 if stats_std is not None else None
        self.input = input
        if(len(channels)>0):
            for i in range(len(stats_mean)):
                if(not(i in channels)):
                    self.stats_mean[:,i]=0
                    self.stats_std[:,i]=1

    def __call__(self, sample):
        datax, labelx, static, static_cat, idxs = sample
        data = datax if self.input else labelx
        #assuming channel last
        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if self.stats_std is not None:
            data = data / self.stats_std
        if self.input:
            return (data, labelx, static, static_cat, idxs)
        else:
            return (datax, data, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return input_shape, label_shape

class NormalizeBatch(TimeSeriesTransform):
    """Normalize using batch statistics."""
    def __init__(self, input=True, channels=[], axis=None):
        super().__init__()
        self.channels = channels
        self.channels_keep = None
        self.input = input
        self.axis = axis

    def __call__(self, sample):
        datax, labelx, static, static_cat, idxs = sample
        data = datax if self.input else labelx
        #assuming channel last
        #batch_mean = np.mean(data,axis=tuple(range(0,len(data)-1)))
        #batch_std = np.std(data,axis=tuple(range(0,len(data)-1)))+1e-8
        batch_mean = np.mean(data,axis=self.axis if self.axis is not None else tuple(range(0,len(data.shape)-1)))
        batch_std = np.std(data,axis=self.axis if self.axis is not None else tuple(range(0,len(data.shape)-1)))+1e-8

        if(len(self.channels)>0):
            if(self.channels_keep is None):
                self.channels_keep = np.setdiff(range(data.shape[-1]),self.channels)

            batch_mean[self.channels_keep]=0
            batch_std[self.channels_keep]=1

        data = (data - batch_mean)/batch_std

        if(self.input):
            return (data, labelx, static, static_cat, idxs)
        else:
            return (datax, data, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return input_shape, label_shape

class ButterFilter(TimeSeriesTransform):
    """Apply filter"""
    def __init__(self, lowcut=50, highcut=50, fs=100, order=5, btype='band', forwardbackward=True, input=True):
        super().__init__()
        self.filter = butter_filter(lowcut, highcut, fs, order, btype)
        self.input = input
        self.forwardbackward = forwardbackward

    def __call__(self, sample):
        datax, labelx, static, static_cat, idxs = sample
        data = datax if self.input else labelx
        if self.forwardbackward:
            data = sosfiltfilt(self.filter, data, axis=0)
        else:
            data = sosfilt(self.filter, data, axis=0)
        if self.input:
            return (data, labelx, static, static_cat, idxs)
        else:
            return (datax, data, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return input_shape, label_shape

class ChannelFilter(TimeSeriesTransform):
    """Select certain channels."""
    def __init__(self, channels=[0], axis=-1, input=True):
        super().__init__()
        self.channels = channels
        self.input = input
        self.axis = axis

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        if(self.input):
            return (np.take(data,self.channels,axis=self.axis), label, static, static_cat, idxs) #(data[...,self.channels], label, static)
        else:
            return (data, np.take(label, self.channels, axis=self.axis), static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        if input_shape is not None:
            shape = list(input_shape)
            axis = self.axis if self.axis >= 0 else len(shape) + self.axis
            shape[axis] = len(self.channels)
            out_signal = tuple(shape)
        else:
            out_signal = None
        return out_signal, label_shape

class Transform(TimeSeriesTransform):
    """Transforms data using a given function i.e. data_new = func(data) for input is True else label_new = func(label)"""
    def __init__(self, func, input=False):
        super().__init__()
        self.func = func
        self.input = input

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        if self.input:
            return (self.func(data), label, static, static_cat, idxs)
        else:
            return (data, self.func(label), static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return None, None

class StaticTransform(TimeSeriesTransform):
    """Transforms static data using a given function i.e. data_new = func(data) for input is True else label_new = func(label)"""
    def __init__(self, func):
        super().__init__()
        self.func = func

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        static, static_cat = self.func(static, static_cat)
        return (data, label, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return input_shape, label_shape

class TupleTransform(TimeSeriesTransform):
    """Transforms data using a given function (operating on both data and label and return a tuple) i.e. data_new, label_new = func(data_old, label_old)"""
    def __init__(self, func):
        super().__init__()
        self.func = func

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        data2, label2 = self.func(data, label, static, static_cat)
        return (data2, label2, static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return None, None

class SequenceToSampleLabelTransform(TimeSeriesTransform):
    """Transforms sequence-level to sample-level labels
    majority vote: pick the most frequent label as segment label (i.e. suitable for 
    single-label classification)
    num_classes: number of output classes
    binary: binary instead of floating point outputs (where the latter represent fractions)
    epoch_length: split the original sequence in ts//epoch_length fragments
    """
    def __init__(self, majority_vote=False, num_classes=2, binary=False, epoch_length=0):
        super().__init__()
        self.majority_vote = majority_vote
        self.num_classes = num_classes
        self.binary = binary
        self.epoch_length = epoch_length

    def __call__(self, sample):
        data, label, static, static_cat, idxs = sample
        
        epoch_length = self.epoch_length if self.epoch_length>0 else len(label)
        if(len(label.shape)==1):#each time step is single-labeled
            label = np.eye(self.num_classes)[label] #now label has shape ts,num_classes
        cnts = np.sum(label.reshape((-1,epoch_length,label.shape[-1])),axis=1)#segments,classes
        if(self.majority_vote):
            label = np.argmax(cnts,axis=-1)#segments
        else:
            if self.binary:
                label = (cnts > 0).astype(np.float32)
            else:
                label = (cnts / epoch_length).astype(np.float32)
        if self.epoch_length > 0:
            return (data, label, static, static_cat, idxs)
        else:#just one segment
            return (data, label[0], static, static_cat, idxs)

    def get_output_shape(self, input_shape, label_shape):
        return input_shape, label_shape
