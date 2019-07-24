import numpy as np

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
