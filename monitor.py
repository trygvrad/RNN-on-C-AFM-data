import tensorflow as tf
import NN
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import copy
import matplotlib
import types
import imageio
def reconfigureCurve(data, std, mean):
    '''
    function that transforms normalized data back to the original parameter embeddingSpace
    data = curve of shape [k,l]
        l lenght of each vector
        k number of vectors
    std = standard deviation
        nympy array of length k or float
    mean = mean
        nympy array of length k or float
    if std and mean are arrays, that means that each vector (k) has a different normalization
    if they are single values, they have a common normalization
    '''
    # make new array for output
    y=np.zeros(data.shape)
    #check if std is an array or not
    if (type(std)==type(np.array([0,0])) and len(std.shape)>0) or  type(std)==type([0,0]):
        #is array, transform
        print(type(std),len(std))
        for i in range(y.shape[-1]):
            for s, j in enumerate(std):
                y[j,i] = data[j,i]*s+mean[j]
    else:
        #is not array, transform
        for i in range(y.shape[-1]):
            y[:,i] = data[:,i]*std+mean
    return y*1000
def save_embedding_map_as_png(path,map):
    if not os.path.isfile(path):
        cmap=matplotlib.cm.get_cmap('viridis')
        t_map=cmap(map/np.max(map))
        print(np.max(map),np.min(map))
        imageio.imwrite(path, t_map)
def add_map(com_var,spes_var):
    '''
    funtion that adds a map to the figure
    input:
    com_var = simpleNamespace that contains variables common for all subplots
    spes_var = simpleNamespace that contains variables related to this subplot
    '''
    spes_var.mapax=com_var.fig.add_axes(spes_var.map_location)
    # plot maps
    pcolor = spes_var.mapax.pcolormesh(np.rot90(spes_var.embeddings))
    spes_var.mapax.invert_yaxis()
    spes_var.mapax.axis('scaled')
    spes_var.mapax.tick_params(
    axis='both',          # changes apply to the both-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,      # ticks along the left edge are off
    right=False,         # ticks along the right edge are off
    labelbottom=False,  # labels along the bottom edge are off
    labelleft=False)  # labels along the left edge are offembeddings = EELScodes.get_activations(EELS_model,


def add_curves_plot(com_var,spes_var, show_bars=True):
    '''
    adds the colored curves and colored bars to the the plot
    '''
    spes_var.ax=com_var.fig.add_axes(spes_var.plot_location)
    ax=spes_var.ax
    if show_bars==True:
        for j, comp2 in enumerate(com_var.comps):
            ax.text(com_var.barxStart+j*com_var.barxStep, com_var.numStart, str(comp2),
                       ha='center')  # number under bars
            if not comp2 == spes_var.comp:
                ax.plot([com_var.barxStart+j*com_var.barxStep]*2,  # plot the black bars
                           [com_var.barStart, com_var.barStart+com_var.barHeight*spes_var.embeddingSpace[0][comp2]/com_var.maxAmp[j]],
                           color=[0, 0, 0, 1])
    numframes=len(spes_var.reconfigured)
    for j,data in enumerate(spes_var.reconfigured):
        # adjust the current embedding
        # nice color gradient from blue to red
        color = com_var.cmap(j/(numframes-1))
        ax.plot(com_var.xax, data,color=color)  # plot the curve
        ax.set_ylim(com_var.ylim)
        ax.set_ylabel(com_var.ylabel)
        ax.set_xlabel(com_var.xlabel)
        ax.set_title(spes_var.title)
        '''ax.set_xticklabels([])
        ax.set_yticklabels([])'''
        # x shift of the colored bars
        if show_bars==True:
            xshift = com_var.barxStep*(-0.25+0.5*j/(numframes-1))
            ax.plot([com_var.barxStart+spes_var.comp_num*com_var.barxStep+xshift]*2,  # plot the colored bars
                       [com_var.barStart, com_var.barStart+com_var.barHeight*j/(numframes-1)],
                       color=color)


def set_bar_locs(com_var,bar_position_parameters={}):
    '''
    the bars that show magnitude are placed at these locations
    bar_position_parameters is a dict with the parameters listed below
        and uses fractional coordinates
    '''
    if not 'bar_y_start' in bar_position_parameters:
        bar_position_parameters['bar_y_start']=0.20
    if not 'bar_num_y_start' in bar_position_parameters:
        bar_position_parameters['bar_num_y_start']=0.16
    if not 'bar_height' in bar_position_parameters:
        bar_position_parameters['bar_height']=0.20
    if not 'bar_x_start' in bar_position_parameters:
        bar_position_parameters['bar_x_start']=0.10
    if not 'bar_x_spread' in bar_position_parameters:
        bar_position_parameters['bar_x_spread']=0.5
    com_var.barStart = com_var.ylim[0]+(com_var.ylim[1]-com_var.ylim[0])*bar_position_parameters['bar_y_start']
    com_var.numStart = com_var.ylim[0]+(com_var.ylim[1]-com_var.ylim[0])*bar_position_parameters['bar_num_y_start']
    com_var.barHeight = (com_var.ylim[1]-com_var.ylim[0])*bar_position_parameters['bar_height']
    com_var.barxStart = com_var.xlim[0]+bar_position_parameters['bar_x_start']*(com_var.xlim[1]-com_var.xlim[0])
    com_var.barxStep = bar_position_parameters['bar_x_spread']*(com_var.xlim[1]-com_var.xlim[0])/len(com_var.comps)

def get_map_loc(colstep,col,rowstep,row,map_position_parameters={}):
    '''
    function that sets the map position based on the input parameters
    '''
    mpp=map_position_parameters
    if not 'map_x_pos' in mpp:
        mpp['map_x_pos']=0.07
    if not 'map_y_pos' in mpp:
        mpp['map_y_pos']=0.47
    if not 'map_width' in mpp:
        mpp['map_width']=0.08*4
    if not 'map_height' in mpp:
        mpp['map_height']=0.4
    return [mpp['map_x_pos']+colstep*col, 1-rowstep*(row+1-mpp['map_y_pos']), mpp['map_width']/4, rowstep*mpp['map_height']]

def plotEmbeddingCurvesSpecial(model, data, imagePath, xlabel='Spectral Dimension', ylabel='Current [pA]',xax=None,std=None,mean=None,use_average=True,plot_only=None,threshold=10**-5,threshold_number=100,show_bars=True, bar_position_parameters={},map_position_parameters={},filename_append=''):
    '''
    function that plots embedding curves, like this:
    #----------------------------------#
    |  #---------#                     |
    |  |         |                     |
    |  |   map   |               curves|
    |  |         |               curves|
    |  #---------#              curves |
    |         bars            curves   |
    |    bars bars bars    curves      |
    |     0    1    2  curves          |
    | curvescurvescurves               |
    #----------------------------------#
    makes one subplot for each embedding
    input:
        model = a trained neural networl
        data = raw data of shape [y,x,l,k] - used for the maps and to calulate average embeddings
            y,x image coordinates
            l lenght of each vector
            k number of vectors in each pixel
        imagePath = current folder and filename for saving
        xlabel = text to put as xlabel, default 'Spectral Dimension'
        ylabel = text to put as ylabel, default 'Amplitude'
        xax = data for the x-axis when plott in a curve, i.e. ax.plot(xax,yax)
        std = standard deviation for transforming data back to original parameter space
        mean = mean for transforming data back to original parameter space
        use_average = if True will generate curves starting from an average of the embeddings,
            if false will generate curves starting with all embeddings at zero
        plot_only = vector to plot, will include all vectors if None
        threshold = the minimum value needed for an embedding to be indluded
        threshold_number = the minimum non-zero pixels in an embedding needed for it to be included
        bar_position_parameters = dict describing position of the bars
        map_position_parameters = dict describing location of map
        filename_append = string to append to end of filename when saved
    '''
    # set the vectors to plot
    if plot_only==None:
        toplot=slice(0,None)
    else:
        toplot=plot_only
    # generate an xax if no xax is provided
    if type(xax)==type(None):
        xax=np.array([range(data.shape[-2])]*range(data.shape[-1])).swapaxes(0,1)
        xax_type_string=''
    else:
        xax_type_string='_CurvesSpecial'
    # generate std and mean if none are provided
    if type(std)==type(None):
        std=1
    if type(mean)==type(None):
        mean=0
    # set up common variables shared between all subplots
    com_var=types.SimpleNamespace()
    com_var.xax=xax[:,toplot]
    com_var.xlabel=xlabel
    com_var.ylabel=ylabel
    # get the embeddings, for maps and average
    embeddings = get_Embeddings(model, imagePath, data)
    numframes = 10
    # calculate average embeddings
    com_var.avgembeddings = np.average(embeddings, axis=(0, 1))
    # get the embeddings to include
    comps = []
    for comp, _ in enumerate(embeddings[0, 0, :]):
        if np.max(embeddings[:, :, comp]) > threshold:
            #if only a few values are nonzero, we assume the embedding fits to
            # outliers and not general trends -> ignore
            if np.sum(embeddings[:, :, comp]>0)>threshold_number:
                comps.append(comp)
            else:
                print('pixels with value over threshold, but still rejected: ',np.sum(embeddings[:, :, comp]>0),' must be at least: ',threshold_number,' to be counted')
    com_var.comps=comps
    # get the range represented in each component
    # the maximum of component i will be compsteps[i]*numframes
    # the minimum will be 0
    compsteps=[np.max(embeddings[:, :, comp])/(numframes-1) for comp in comps]
    # make the curves
    #     first set the embedding space
    #     do this for all subfigures, so you can decode them in prarallel
    if use_average==True:
        embeddingSpace = np.array([com_var.avgembeddings]*len(comps)*numframes)
    else:
        embeddingSpace = np.array([[0.0]*len(com_var.avgembeddings)]*len(comps)*numframes)
    for i, comp in enumerate(comps):
        for frame in range(numframes):
            embeddingSpace[i*numframes+frame, comp] = compsteps[i]*frame
    #     decode the embeddings to generate the curves
    reconstructed = model.get_decoded(embeddingSpace)
    reconstructed = reconstructed.reshape(
        (len(comps), numframes, data.shape[-2], data.shape[-1]))
    # reconfigure curves back to original parameter space and get max and min ylimits
    ylim_min=np.inf
    ylim_max=float()-np.inf
    spes_var=[]
    for i, comp in enumerate(comps):
        spes_var.append(types.SimpleNamespace())
        spes_var[i].reconfigured=[]
        spes_var[i].embeddingSpace=[]
        for frame in range(numframes):
            spes_var[i].reconfigured.append(reconfigureCurve(
                reconstructed[i, frame], std, mean)[:,toplot])
            ylim_min=np.min((ylim_min,np.min((spes_var[i].reconfigured[-1]))))
            ylim_max=np.max((ylim_max,np.max((spes_var[i].reconfigured[-1]))))
            spes_var[i].embeddingSpace.append(embeddingSpace[i*numframes+frame])
    # calculate number of rows in the figure
    # and get the maxmum amplitude in each component
    #     used to normalize the height of the bars
    rows = (len(comps)+3)//4
    rowstep = 1/rows
    com_var.maxAmp = []
    for i,comp in enumerate(comps):
        spes_var[i].comp=comp
        spes_var[i].comp_num=i
        row = i//4
        col = i % 4
        #magic expression for making the map in subfigure
        colstep=0.24
        spes_var[i].map_location=(get_map_loc(colstep,col,rowstep,row,map_position_parameters=map_position_parameters))
        #magic expression for placing the subfigures
        spes_var[i].plot_location=([0.06+colstep*col, 1-rowstep*(row+1-0.15), 0.18, rowstep*(1-0.28)])
        spes_var[i].embeddings=embeddings[:,:,comp]
        spes_var[i].title='Embedding '+str(i+1)
        #save_embedding_map_as_png(imagePath+'embeddings_'+str(i)+'.png',spes_var[i].embeddings)
        # get the maximum amplitude
        com_var.maxAmp.append(np.max(embeddings[:, :, comp]))
    # declare the xlim and ylim to be used in the subfigures
    com_var.xlim = [np.min(xax), np.max(xax)]
    com_var.ylim = [ylim_min-(ylim_max-ylim_min)*0.05, ylim_max+(ylim_max-ylim_min)*0.05]
    # set up the locations for bars based on xlim and ylim
    set_bar_locs(com_var,bar_position_parameters=bar_position_parameters)
    #set the colormap for the curves/bars
    com_var.cmap = matplotlib.cm.get_cmap('viridis')
    #create the figure
    com_var.fig = plt.figure(figsize=(16, 4*rows),dpi=200)
    # make plot for each embedding
    for i, comp in tqdm.tqdm(enumerate(comps), total=len(comps), desc="Embeddings plotted"):
        # plot curves
        add_curves_plot(com_var,spes_var[i],show_bars=show_bars)
        # plot the map
        add_map(com_var,spes_var[i])
    time.sleep(0.001)
    # save the figure
    save_path=imagePath+xax_type_string
    if use_average==False:
        save_path=save_path+'_FromZero'
    if not plot_only==None:
        save_path=save_path+'_'+str(toplot)
        #com_var.fig.savefig(save_path+'.svg')
    com_var.fig.savefig(save_path+filename_append+'.png')
    # close the figure
    com_var.fig.clf()
    plt.close(com_var.fig)

def get_Embeddings(model, imagePath, data):
    '''
    function that uses trained model to calculate embeddings and saves them
    embedding maps
    input:
    model = the trained neural network
    data = raw data of shape [y,x,l,k]
        y,x image coordinates
        l lenght of each vector
        k number of vectors in each pixel
    imagePath = active folder
    '''
    # if already run, load old values
    if os.path.isfile(imagePath+'embeddings.npy'):
        embeddings = np.load(imagePath+'embeddings.npy')
    else:
        # calculate embeddings
        # the network requires the data to be [x*y,l,k] not [x,y,l,k], so we reshape
        embeddings = model.get_embeddings(
            data.reshape((-1, data.shape[-2], data.shape[-1])), batch_size=1000)
        embeddings = embeddings.reshape((data.shape[0], data.shape[1], -1))
        # save calculated embeddings
        np.save(imagePath+'embeddings', embeddings)

    return embeddings


def plotEmbeddings(model, data, imagePath,threshold=10**-5,threshold_number=100):
    '''
    function that uses trained model to calculate embeddings and plot
    embedding maps
    input:
    model = the trained neural network
    data = raw data of shape [y,x,l,k]
        y,x image coordinates
        l lenght of each vector
        k number of vectors in each pixel
    imagePath = the folder to save the plot
    threshold = the minimum value needed for an embedding to be indluded
    threshold_number = the minimum non-zero pixels in an embedding needed for it to be included
    '''
    #get the embeddings from the trained model
    embeddings = get_Embeddings(model, imagePath, data)
    # plot only embeddings that have a maximum value above threshold
    comps = []
    for comp, _ in enumerate(embeddings[0, 0, :]):
        if np.max(embeddings[:, :, comp]) > threshold:
            #if only a few values are nonzero, we assume the embedding fits to
            # outliers and not general trends -> ignore
            if np.sum(embeddings[:, :, comp]>0)>threshold_number:
                comps.append(comp)
            else:
                print(np.sum(embeddings[:, :, comp]>0))
    #calculate number of rows needed
    rows = (len(comps)+3)//4
    # set up the figure
    fig = plt.figure(figsize=(16, 4*rows))
    ax = []
    for i, comp in tqdm.tqdm(enumerate(comps), total=len(comps), desc="Maps plotted"):
        row = i//4
        col = i % 4
        rowstep = 1/rows
        # make axes
        # magical expressions for placement
        ax.append(fig.add_axes([0.06+0.24*col, 1-rowstep *
                                (row+1-0.15), 0.18, rowstep*(1-0.28)]))
        # plot the embeddings
        pcolor = ax[-1].pcolormesh(embeddings[:, :, comp])
        # pcolor.set_clim(0, 0.001)
        ax[-1].invert_yaxis()
        fig.colorbar(pcolor, ax=ax[-1])
        ax[-1].axis('scaled')
        ax[-1].set_title('Embedding '+str(comp))
    #save the figure
    fig.savefig(imagePath+'embeddings.png')
    fig.clf()
    plt.close(fig)
    print('made Maps', imagePath+'embeddings.png')


def plotLoss(imagePath):
    '''
    Plots the evolution of the loss as a function of time
    Gets the evolution of loss from the filename in stored weights
    Should probably be changed to read log.csv instead
    '''
    #get folder
    folder = imagePath.strip(imagePath.split('/')[-1])
    epochs = []
    losses = []
    #list all files in folder
    files = os.listdir(folder)
    #sort all files in folder on time
    files.sort(key=lambda x: os.path.getmtime(folder+x))
    for file in files:
        if '.hdf5' in file:
            try:
                #get loss and epoch information from the filename
                epochs.append(int(file.split('-')[0].split('.')[1]))
                losses.append(float(file.split('-')[1].strip('.hdf5')))
            except:
                print('epoch and loss not found in filename '+file)
    #make and save plot
    fig = plt.figure(figsize=(4, 4))
    ax = [fig.add_subplot(1, 1, 1)]
    ax[0].semilogx(epochs, losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(folder+'loss.png')
    fig.clf()
    plt.close(fig)
    print('made loss', folder+'loss.png')


def plotLossMap(model, data, imagePath):
    '''
    function that uses model to predict data from data and calculates the loss
    in each pixel of the image
    Plots a loss map first for all vectors together
    Then for each vector independently
    input:
    model = the trained neural network
    data = raw data of shape [y,x,l,k]
        y,x image coordinates
        l lenght of each vector
        k number of vectors in each pixel
    imagePath = the folder to save the plot
    '''
    # use the model to predict the data from data
    predicted = model.predict(
        data.reshape(-1, data.shape[-2], data.shape[-1]))
    predicted = predicted.reshape(data.shape)
    # save the preicted numpy array
    np.save(imagePath+'predicted', predicted)
    #calculate the loss
    mse = ((data - predicted)**2).mean(axis=2)
    # save the loss as a numpy array
    np.save(imagePath+'loss', mse)
    #calculate the total loss
    mseTot = np.sum(mse, axis=2)
    # set up the figure
    #calculate teh number of rows in the figure
    rows = (data.shape[-1]+1+3)//4
    rowstep = 1/rows
    #make the figure
    fig = plt.figure(figsize=(16, 4*rows))
    ax = []  # axes for curves
    # magical expression for placement of axes in the figure
    ax.append(fig.add_axes([0.06, 1-rowstep *
                            (1-0.15), 0.18, rowstep*(1-0.28)]))
    #plot the total loss data
    pcolor = ax[-1].pcolormesh(mseTot[:, :])
    #invert yaxis because the data is [y,x] and not [x,y]
    ax[-1].invert_yaxis()
    # add the colorbar
    fig.colorbar(pcolor, ax=ax[-1])
    # set scaled so that the size of each pixel is the same in the x and y direction
    ax[-1].axis('scaled')
    ax[-1].set_title('Total Loss')
    # repeat for each Vector
    for i in range(data.shape[-1]):
        row = (i+1)//4
        col = (i+1) % 4
        ax.append(fig.add_axes([0.06+0.24*col, 1-rowstep *
                                (row+1-0.15), 0.18, rowstep*(1-0.28)]))
        # plot loss
        pcolor = ax[-1].pcolormesh(mse[:, :, i])
        # finalize figure
        ax[-1].invert_yaxis()
        fig.colorbar(pcolor, ax=ax[-1])
        ax[-1].axis('scaled')
        ax[-1].set_title('Vector '+str(i))
    # save the figure
    fig.savefig(imagePath+'Lossmap.png')
    fig.clf()
    plt.close(fig)
    print('made Maps', imagePath)


def NNplot(load_path, weights, data, xax, std, mean):
    imagePath = weights.strip('.hdf5')
    print(imagePath)
    #check if these weights have already been run, assume that it has been run if imagePath+'embeddings.npy' exists
    if not os.path.isfile(imagePath+'embeddings.npy'):
        #load the model
        model = NN.NN(load_path=load_path, weights=weights)
        for i in range(data.shape[-1]):
            #plot individual curves for each vector with change from average
            plotEmbeddingCurvesSpecial(model, data, imagePath,plot_only=i,ylabel='Current [pA]', xlabel='Voltage', xax=xax, std=std, mean=mean)
            #plot individual curves for each vector with change from zero
            plotEmbeddingCurvesSpecial(model, data, imagePath,plot_only=i,use_average=False,ylabel='Current [pA]', xlabel='Voltage', xax=xax, std=std, mean=mean)
        #plot all curves together with change from average
        plotEmbeddingCurvesSpecial(model, data, imagePath, xlabel='Voltage', xax=xax, std=std, mean=mean)
        #plot all curves together with change from zero
        plotEmbeddingCurvesSpecial(model, data, imagePath, use_average=False,ylabel='Current [pA]', xlabel='Voltage', xax=xax, std=std, mean=mean)
        #include all embeddings (do not use threshold to select)
        plotEmbeddingCurvesSpecial(model, data, imagePath,threshold=-1,threshold_number=-1, ylabel='Current [pA]', xlabel='Voltage', xax=xax, std=std, mean=mean,filename_append='_no_threshold')
        #plot loss maps
        plotLossMap(model, data, imagePath)
        #plot embedding maps
        plotEmbeddings(model, data, imagePath)
        #plot loss maps
        plotLoss(imagePath)
        #plot final figure
        plotEmbeddingCurvesSpecial(model, data, imagePath, plot_only=1,ylabel='Current [pA]', xlabel='Voltage [V]', xax=xax, std=std, mean=mean,
            show_bars=False,
            bar_position_parameters={'bar_y_start':0.2,'bar_num_y_start':0.16,'bar_height':0.2,'bar_x_start':0.1,'bar_x_spread':0.5},
            map_position_parameters={'map_x_pos':0.07,'map_y_pos':0.32,'map_width':0.115*4,'map_height':0.55},
            filename_append='_final')
        reset_keras()
    else:
        print('already found '+imagePath)


def runOnFolder(path, data, xax, std, mean):
    # get all files in 'path'
    files = os.listdir(path)
    # sort them on time changed
    files.sort(key=lambda x: os.path.getmtime(path+'/'+x))
    load_path = None
    weights = None
    # look for trained models in this folder
    for file in files:
        if '.h5' in file:
            load_path = path+'/'+file
        elif '.hdf5' in file:
            weights = path+'/'+file
    # if there are any trained models in this folder, weights should now be the most recent weights
    if (not load_path is None) and (not weights is None):
        # make plots from model
        NNplot(load_path, weights, data, xax, std, mean)

    # run recursively on subfolders
    for file in files:
        if os.path.isdir(path+'/'+file):
            runOnFolder(path+'/'+file, data, xax, std, mean)


def monitor(path, data, xax, std, mean, sleeptime=3600):
    while True:
        # run a pass on the folder 'path' and all subfolders
        runOnFolder(path, data, xax, std, mean)
        # sleep for an sleeptime repeating
        time.sleep(sleeptime)

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import gc

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    #gc.collect()

#run the monitor
def main():
    if __name__== "__main__" :
        #################### this section contains the configuration that is run when monitor.py is run as a script
        # set what graphics card to use
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # set to allow dynamic allocation of memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        #the config for the monitor
        path = 'models'
        data = np.load('prosessedY.npy')
        xax = np.load('prosessedX.npy')
        std = np.load('std.npy')
        mean = np.load('mean.npy')
        monitor(path, data, xax, std, mean, sleeptime=3600)
main()
