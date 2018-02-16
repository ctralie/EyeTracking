import numpy as np
import scipy.io as sio
import scipy.signal
import matplotlib.pyplot as plt
from MergeTree import *

def fillNaNs(x):
    """
    Linearly interpolate between endpoints on regions of NaNs
    (assumes first and last point are not NaNs)
    """
    y = np.array(x)
    (IN_NORMAL, IN_NAN) = (0, 1)
    state = IN_NORMAL
    starti = 0
    for i in range(1, len(x)-1):
        if state == IN_NORMAL and np.isnan(x[i]):
            starti = i-1
            state = IN_NAN
        elif state == IN_NAN and (not np.isnan(x[i])):
            endi = i+1
            y[starti+1:endi] = np.linspace(x[starti], y[endi], i-starti)
            state = IN_NORMAL 
    #Take care of beginning/end
    y = y[np.isnan(y) == False]
    return y

def getAllTrajectories(medk = 51, doFillNaNs = True, trange = None, missingRatio = 0.2):
    """
    Load in all eye tracker trajectories and clean up
    :param medk: Length of median filter (default 51)
    :param doFillNaNs: Whether to fill in NaN regions using linear interpolation
    :param trange: If specified, only include this subset of the data
    :param missingRatio: Discard trajectories with more than this proportion\
        of missing data
    :returns (Xs, Scores): Xs is an array of flat numpy arrays of trajectories,\
        sorted in increasing order of score.  Scores is the corresponding array\
        of scores
    """
    D = sio.loadmat('dataset.mat')['data'][0]
    #First pull out all of the scores
    scores = [D[i]['score'][0][0].flatten()*1.0 for i in range(D.size)]
    scores = np.array(scores).flatten()
    #Pull out the trajectories with the lowest scores first
    idx = np.argsort(scores)
    #Now pull out all of the trajectories
    Xs = []
    for i in range(len(scores)):
        lx = D[idx[i]]['lx'][0, 0].flatten()
        if trange:
            lx = lx[trange[0]:trange[1]]
            if np.sum(np.isnan(lx)) > 0:
                scores[idx[i]] = -1
                continue
        if np.sum(np.isnan(lx))/float(lx.size) > missingRatio:
            scores[idx[i]] = -1
            continue
        if doFillNaNs:
            lx = fillNaNs(lx)
        lx = scipy.signal.medfilt(lx, medk)
        Xs.append(lx)
    scores = scores[idx]
    scores = scores[scores >= 0]
    return (Xs, scores)

def plotAllXTrajectories(medk = 51, doFillNaNs = True):
    """
    Plot the trajectories all on top of each other for visualization
    """
    (Xs, scores) = getAllTrajectories(medk, doFillNaNs)
    res=10
    plt.figure(figsize=(res, res*len(scores)/10))

    for i in range(len(scores)):
        plt.subplot(len(scores), 1, i+1)
        plt.plot(Xs[i])
        plt.title("%i, Score = %g"%(i, scores[i]))
    
if __name__ == '__main__':
    medk = 51
    plotTrajectories = False
    if plotTrajectories:
        plotAllXTrajectories(medk, doFillNaNs = False)
        plt.savefig("AllTrajectories.svg", bbox_inches = 'tight')
        plt.clf()
        plotAllXTrajectories(medk, doFillNaNs = True)
        plt.savefig("AllTrajectories_FilledIn.svg", bbox_inches = 'tight')
    (Xs, scores) = getAllTrajectories(medk, doFillNaNs = False, trange = [3800, 4600])

    #Compute persistence diagrams and sum over persistence images
    AllPersistences = np.zeros(len(scores))
    #Maximum persistence to consider (if -1, consider all persistences)
    maxPers = -1
    res = 0.1 #Resolution of persistence image
    for i, x in enumerate(Xs):
        print("Computing DGM %i..."%i)
        #Sum persistences going up and down
        (MT, PS, I) = mergeTreeFrom1DTimeSeries(x)
        persup = I[:, 1]-I[:, 0]
        (MT, PS, I) = mergeTreeFrom1DTimeSeries(-x)
        persdown = I[:, 1]-I[:, 0]
        pers = np.array(persup.tolist() + persdown.tolist())
        if maxPers > -1:
            pers = pers[pers <= maxPers]
        AllPersistences[i] = np.sum(pers)
    
    #Now re-sort the signals by the Fiedler vector of the graph
    #Laplacian built over Euclidean distance between the histograms
    plt.clf()
    plotres=10
    plt.figure(figsize=(plotres, plotres*len(scores)/3))
    idxs = np.argsort(AllPersistences)
    for i, idx in enumerate(idxs):
        plt.subplot(len(scores), 1, 1+i)
        plt.plot(Xs[idx])
        plt.title("Score = %g, TotalPersistence = %.3g"%(scores[idx], AllPersistences[idx]))
    plt.savefig("AllTrajectories_Resorted.svg", bbox_inches = 'tight')