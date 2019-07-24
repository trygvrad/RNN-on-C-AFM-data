
from scipy import interpolate
from scipy.signal import savgol_filter as sg
import scipy
import datetime
import numpy as np


def subtractExpBackground(data, xrange=None):
    data2 = np.float64(np.copy(data))
    x = range(data.shape[2])
    if type(xrange) == type(None):
        xrange = x
    p0 = [4.19082741e+02, -1.93625569e-03]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            popt, pcov = scipy.optimize.curve_fit(
                scaledExp, xrange, data2[i, j, xrange], p0=p0)
            data2[i, j] = data2[i, j]-scaledExp(x, popt[0], popt[1])
            # print(popt)
    return data2


def scaledExp(x, a, b):
    return a*np.exp((np.array(x))*b)


#!python numbers=enable
# https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
def sgolay2d(z, window_length=5, polyorder=3, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = (polyorder + 1) * (polyorder + 2) / 2.0

    if window_length % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_length**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_length // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k-n, n) for k in range(polyorder+1) for n in range(k+1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat(ind, window_length)
    dy = np.tile(ind, [window_length, 1]).reshape(window_length**2, )

    # build matrix of system of equation
    A = np.empty((window_length**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros((new_shape))
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - \
        np.abs(np.flipud(z[1:half_size+1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + \
        np.abs(np.flipud(z[-half_size-1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - \
        np.abs(np.fliplr(z[:, 1:half_size+1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + \
        np.abs(np.fliplr(z[:, -half_size-1:-1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - \
        np.abs(np.flipud(np.fliplr(z[1:half_size+1, 1:half_size+1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + \
        np.abs(
            np.flipud(np.fliplr(z[-half_size-1:-1, -half_size-1:-1])) - band)

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - \
        np.abs(np.flipud(Z[half_size+1:2*half_size+1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - \
        np.abs(np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band)

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_length, -1))
        r = np.linalg.pinv(A)[2].reshape((window_length, -1))
        return scipy.signal.fftconvolve(Z, -r, mode='valid'), scipy.signal.fftconvolve(Z, -c, mode='valid')


def normalize(data, data_normal=None, extra_output=None):
    """
    Normalizes the data

    Parameters
    ----------
    data : numpy, array
        data to normalize
    data_normal : numpy, (optional)
        data set to normalize with

    Returns
    -------
    data_norm : numpy, array
        Output of normalized data
    """

    if data_normal is None:
        data_norm = np.float64(np.copy(data))
        mean = np.mean(np.float64(data_norm.reshape(-1)))
        data_norm -= mean
        std = np.std(data_norm)
        data_norm /= std
    else:
        data_norm = np.float64(np.copy(data))
        mean = np.mean(np.float64(data_normal.reshape(-1)))
        data_norm -= mean
        std = np.std(data_normal)
        data_norm /= std
    if extra_output == None:
        return data_norm
    else:
        return data_norm, std, mean

####################################################################################################################
#####################################   Savitzky-Golay filter   ####################################################
## from https://github.com/jagar2/Revealing-Ferroelectric-Switching-Character-Using-Deep-Recurrent-Neural-Networks #
####################################################################################################################


#import codes.processing.filters
#data.I=codes.processing.filters.savgol(np.float64(np.copy(data.I)), num_to_remove=3, window_length=5, polyorder=3,fit_type='linear')


def savgol(data_, num_to_remove=3, window_length=7, polyorder=3, fit_type='spline'):
    """
    Applies a Savitzky-Golay filter to the data which is used to remove outlier or noisy points from the data

    Parameters
    ----------
    data_ : numpy, array
        array of loops
    num_to_remove : numpy, int
        sets the number of points to remove
    window_length : numpy, int
        sets the size of the window for the sg filter
    polyorder : numpy, int
        sets the order of the sg filter
    fit_type : string
        selection of type of function for interpolation

    Returns
    -------
    cleaned_data : numpy array
        array of loops
    """
    data = np.copy(data_)

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    cleaned_data = np.copy(data)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                sg_ = sg(data[i, j, :, k],
                         window_length=window_length, polyorder=polyorder)
                diff = np.abs(data[i, j, :, k] - sg_)
                sort_ind = np.argsort(diff)
                remove = sort_ind[-1 * num_to_remove::].astype(int)
                cleaned_data[i, j, remove, k] = np.nan

    # clean and interpolates data
    cleaned_data = clean_interpolate(cleaned_data, fit_type)

    return cleaned_data


def interpolate_missing_points(data, fit_type='spline'):
    """
    Interpolates bad pixels in piezoelectric hysteresis loops.\n
    The interpolation of missing points allows for machine learning operations

    Parameters
    ----------
    data : numpy array
        array of loops
    fit_type : string (optional)
        selection of type of function for interpolation

    Returns
    -------
    data_cleaned : numpy array
        array of loops
    """

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                if any(~np.isfinite(data[i, j, :, k])):

                    # selects the index where values are nan
                    ind = np.where(np.isnan(data[i, j, :, k]))

                    # if the first value is 0 copies the second value
                    if 0 in np.asarray(ind):
                        data[i, j, 0, k] = data[i, j, 1, k]

                    # selects the values that are not nan
                    true_ind = np.where(~np.isnan(data[i, j, :, k]))

                    # for a spline fit
                    if fit_type == 'spline':
                        # does spline interpolation
                        spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                                          data[i, j, true_ind, k].squeeze())
                        data[i, j, ind, k] = spline(point_values[ind])

                    # for a linear fit
                    elif fit_type == 'linear':

                        # does linear interpolation
                        data[i, j, :, k] = np.interp(point_values,
                                                     point_values[true_ind],
                                                     data[i, j, true_ind, k].squeeze())

    return data.squeeze()


def clean_interpolate(data, fit_type='spline'):
    """
    Function which removes bad data points

    Parameters
    ----------
    data : numpy, float
        data to clean
    fit_type : string  (optional)
        sets the type of fitting to use

    Returns
    -------
    data : numpy, float
        cleaned data
    """

    # sets all non finite values to nan
    data[~np.isfinite(data)] = np.nan
    # function to interpolate missing points
    data = interpolate_missing_points(data, fit_type)
    # reshapes data to a consistent size
    data = data.reshape(-1, data.shape[2])
    return data


def reduce_dimensionality(dataMat, bins, extra_output=False):
    shape = dataMat.shape
    print(shape)
    redDataMat = []
    numV = shape[-2]
    binsize = numV//bins
    startCutoff = numV-bins*binsize
    if len(shape) > 3:
        for i in range(shape[0]):
            redDataMat.append([])
            for j in range(shape[1]):
                redDataMat[-1].append([])
                for k in range(binsize+startCutoff, numV+1, binsize):
                    redDataMat[-1][-1].append(np.average(dataMat[i,
                                                                 j, k-binsize:k], axis=0))
    elif len(shape) == 2:
        for k in range(binsize+startCutoff, numV+1, binsize):
            redDataMat.append(np.average(dataMat[
                k-binsize:k], axis=0))

    print('Averaging to '+str(bins)+' on rising and falling end')
    print('Averaging over '+str(binsize)+' datapoints')
    print('cutting '+str(startCutoff)+' datapoint on each end')
    redDataMat = np.array(redDataMat)
    if extra_output == True:
        return redDataMat, binsize, startCutoff
    else:
        return redDataMat
