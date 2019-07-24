
import numpy as np
import xlrd
import types
import NN.preprocessing
'''
read the xls file
'''
wb = xlrd.open_workbook('_I_(BV)_1 (1).xls')
sheet1 = wb.sheet_by_index(0)


arr = []
for row in range(1, sheet1.nrows):
    arr.append([])
    for col in range(sheet1.ncols):
        arr[-1].append(float(sheet1.cell(row, col).value))
arr = np.array(arr)
arr = arr.swapaxes(0, 1)
current = []
for i in range(40):
    current.append([])
    for j in range(40):
        current[-1].append([])
        for k in range(arr.shape[1]):
            current[-1][-1].append([arr[(i*40+j)*4+1, k], arr[(i*40+j)*4+3, k]])

current = np.array(current)
'''
prepare data for the neural net
'''
bins = 75
data = types.SimpleNamespace()
data.I = current
data.V = arr[0]
data.V = np.swapaxes([-data.V[::-1], -data.V], 0, 1)
# reduce dimensionality
data.IShort, binsize, startCutoff = NN.preprocessing.reduce_dimensionality(
    data.I, bins, extra_output=True)
data.VShort = NN.preprocessing.reduce_dimensionality(data.V, bins)
# normalize
data.INorm, std, mean = NN.preprocessing.normalize(
    data.IShort, extra_output=True)
#save
trainingforML = np.array(data.INorm)
np.save('std', std)
mean=mean-51.32 # The current has an offset of 51.32 pA 
np.save('mean', mean)
np.save('binsize', binsize)
np.save('prosessedY', trainingforML)
data.VShort=data.VShort*15 # a 15X amplifier was used for the measurement, i.e. the range goes from 0 to 15 V
np.save('prosessedX', data.VShort)
print('preprocessing done')
