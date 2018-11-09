import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
import matplotlib.pyplot as plt
from GSPLib.GreedyPerm import *
from GSPLib.DGMTools import *
from GSPLib.CSMSSMTools import *
from Geom3D.PolyMesh import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import glob
import os
import pandas
import subprocess
import ripser.ripser as ripser
from sklearn.linear_model import Ridge, RidgeCV

def convertGUDHIPD(pers):
    Is = [[], [], []]
    for i in range(len(pers)):
        (dim, (b, d)) = pers[i]
        Is[dim].append([b, d])
    #Put onto diameter scale so it matches rips more closely
    return [2*np.sqrt(np.array(I)) for I in Is]

def getPlotName(f, ext):
    s = f.split("qrtosso_")[1]
    s1 = s[0:2]
    s2 = s.split("rec_conv_")[1]
    s2 = s2.split(ext)[0]
    s = "%s_%s"%(s1, s2)
    return s

def getPIs(files, useH1, useH2, psigma, weightfn = lambda b, l: l*(l > 0.05)):
    #Compute persistence images
    PIs = []
    N = len(files)
    for i in range(N):
        print("Getting persistence images %i of %i"%(i+1, N))
        X = sio.loadmat(files[i])
        [I0, I1, I2] = [X['I0'], X['I1'], X['I2']]
        I = np.array([])
        if useH1:
            res = getPersistenceImage(I1, [0, 1, 0, 1], 0.01, weightfn = weightfn, psigma = psigma)
            I = res['PI']
        if useH2:
            res = getPersistenceImage(I2, [0, 1, 0, 1], 0.01, weightfn = weightfn, psigma = psigma)
            if useH1:
                I = np.concatenate((I, res['PI']), 0)
            else:
                I = res['PI']
        PIs.append(I.flatten())
    return (res, np.array(PIs))


def ridgeRegressionPIs(files, useH1 = True, useH2 = True, psigma=0.1):
    savestr = ""
    if useH1:
        savestr += "H1"
    if useH2:
        savestr += "H2"
    N = len(files)
    (res, PIs) = getPIs(files, useH1, useH2, psigma)
    TrabNums = pandas.read_csv("TrabNums.csv")
    specimens = [s.lower() for s in TrabNums['Specimen'].values.tolist()]
    values = TrabNums['Trab num. (1/mm)'].values
    specimen2val = {specimens[i]:values[i] for i in range(len(specimens))}
    x = [specimen2val[f.split("qrtosso_")[1][0:2]] for f in files]
    x = np.array(x)
    alphas = 10.0**np.linspace(-3, 3, 1000)
    clf = RidgeCV(alphas=alphas, store_cv_values=True).fit(PIs, x)
    errs = np.sqrt(np.mean(clf.cv_values_, 0)) #RMSE
    idx = np.argmin(errs)
    alpha = alphas[idx]
    alpha = 0.08

    plt.figure()
    plt.semilogx(alphas, errs)
    plt.stem([alpha], [errs[idx]])
    ry = np.max(errs) - np.min(errs)
    plt.ylim([np.min(errs)-0.1*ry, np.max(errs)+0.1*ry])
    plt.xlabel("$\\alpha$")
    plt.ylabel("RMSE")
    plt.title("$\\alpha = %.3g, err = %.3g$"%(alpha, errs[idx]))
    plt.savefig("CrossVal_%s.svg"%savestr, bbox_inches='tight')

    clf = Ridge(alpha=alpha).fit(PIs, x)
    w = clf.coef_
    if useH1 and useH2:
        dim = int(np.sqrt(PIs.shape[1]/2))
        w = np.reshape(w, (dim*2, dim))
        I1 = w[0:dim, :]
        I2 = w[dim::, :]
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        m = np.max(np.abs(I1))
        plt.imshow(I1, vmin = -m, vmax = m, cmap = 'coolwarm', extent = (res['xr'][0], res['xr'][-1], res['yr'][-1], res['yr'][0]))
        plt.gca().invert_yaxis()
        plt.title("H1")
        plt.xlabel("Birth Time")
        plt.ylabel("Lifetime")
        plt.subplot(122)
        m = np.max(np.abs(I2))
        plt.imshow(I2, vmin = -m, vmax = m, cmap = 'coolwarm', extent = (res['xr'][0], res['xr'][-1], res['yr'][-1], res['yr'][0]))
        plt.gca().invert_yaxis()
        plt.title("H2")
        plt.xlabel("Birth Time")
        plt.ylabel("Lifetime")
    else:
        dim = int(np.sqrt(PIs.shape[1]))
        w = np.reshape(w, (dim, dim))
        plt.figure(figsize=(6, 6))
        m = np.max(np.abs(w))
        plt.imshow(w, vmin = -m, vmax = m, cmap = 'coolwarm', extent = (res['xr'][0], res['xr'][-1], res['yr'][-1], res['yr'][0]))
        plt.gca().invert_yaxis()
        plt.title(savestr)
        plt.xlabel("Birth Time")
        plt.ylabel("Lifetime")
    plt.savefig("RidgeCoeffs_%s.png"%savestr, bbox_inches='tight')


def comparePersistenceImages(files, useH1 = True, useH2 = True, psigma = 0.1):
    N = len(files)
    #Make plot names
    plotNames = []
    for f in files:
        plotNames.append(getPlotName(f, "_Is.mat"))
    (res, PIs) = getPIs(files, useH1, useH2, psigma)


    #Compute pairwise Euclidean distances between 
    #persistence images, and plot results
    D = getSSM(PIs)
    w, v = linalg.eigh(D)
    idx = np.argsort(v[:, 0])
    D = D[idx, :]
    D = D[:, idx]
    plotNames = [plotNames[i] for i in idx]
    x = np.arange(D.shape[0])
    plt.figure(figsize=(10, 10))
    plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
    plt.xticks(x, plotNames, rotation='vertical')
    plt.yticks(x, plotNames, rotation='horizontal')
    plt.colorbar()
    plt.title("Persistence Image Bone Similarity")
    plt.savefig("PISimilarity.svg", bbox_inches = 'tight')

    clusters = [[0, 8], [9, 10], [11, 14], [15, 17]]
    for i in range(len(clusters)):
        [i1, i2] = clusters[i]
        os.mkdir("Cluster%i"%i)
        for k in range(i1, i2+1):
            subprocess.call(["cp", "PDs/%s.png"%plotNames[k], "Cluster%i"%i])


def getBonePDs(filename, M = -1, doAlpha = True, useGUDHI = True, psigma = 0.1):
    matfilename = "%s_Is.mat"%filename[0:-4]

    #Step 1: Load mesh
    tic = time.time()
    filetemp = "%s.off"%filename[0:-4]
    subprocess.call(["meshlabserver", "-i", filename, "-s", "removedup.mlx", "-o", filetemp])
    (X, VColors, ITris) = loadOffFileExternal(filetemp)
    print("Elapsed Time Loading: %g"%(time.time() - tic))
    
    #Step 2: Do greedy permutation if requested
    if M > -1:
        tic = time.time()
        res = getGreedyPerm(X, M, Verbose = True)
        Y = res['Y']
        thresh = res['lambdas'][1]
        scale = res['lambdas'][-1]
        print("thresh = %g"%thresh)
        print("scale = %g"%scale)
        print("Elapsed Time Random Perm: %g"%(time.time() - tic))
    else:
        thresh = np.inf
        scale = 0.0
        Y = X
    Y = Y + 0.001*np.random.randn(Y.shape[0], Y.shape[1])

    #Step 3: Compute Alpha or Rips complex
    if os.path.exists(matfilename):
        #Load precomputed persistence diagrams
        X = sio.loadmat(matfilename)
        [I0, I1, I2] = [X['I0'], X['I1'], X['I2']]
        Is = [I0, I1, I2]
        print("Skipping persistence computation, using precomputed diagrams from %s.."%matfilename)
    else:
        tic = time.time()
        if doAlpha:
            if useGUDHI:
                import gudhi
                alpha_complex = gudhi.AlphaComplex(points = Y.tolist())
                simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square = np.inf)
                print("Number of simplices = %i"%simplex_tree.num_simplices())
                pers = simplex_tree.persistence()
                Is = convertGUDHIPD(pers)
            else:
                from dionysus import Filtration, StaticPersistence, fill_alpha3D_complex
                f = Filtration()
                fill_alpha3D_complex(Y.tolist(), f)
                p = StaticPersistence(f)
                print("StaticPersistence initialized")
                p.pair_simplices()
                print("Simplices paired")
                smap = p.make_simplex_map(f)
                Is = [[], [], [], []]
                for i in p:
                    if i.sign():
                        b = smap[i]
                        birth = np.sqrt(b.data[0])
                        death = np.inf
                        if not i.unpaired():
                            d = smap[i.pair()]
                            death = np.sqrt(d.data[0])
                        Is[b.dimension()].append([birth, death])
                Is = [np.array(I) for I in Is]                    
        else:
            Is = ripser.doRipsFiltration(Y, 2)
        print("Elapsed Time Persistence: %g"%(time.time() - tic))

    #Compute persistence images
    tic = time.time()
    weightfn = lambda b, l: l*(l > 0.05)
    res1 = getPersistenceImage(Is[1], [0, 2, 0, 2], 0.01, weightfn = weightfn, psigma = psigma)
    res2 = getPersistenceImage(Is[2], [0, 2, 0, 2], 0.01, weightfn = weightfn, psigma = psigma)
    print("\n\n\nElapsed Time Persistence Images: %g\n\n\n"%(time.time() - tic))
    sio.savemat(matfilename, {"I0":Is[0], "I1":Is[1], "I2":Is[2], "PI1":res1['PI'], "PI2":res2['PI']})
    
    #Do plots
    plt.clf()
    plt.subplot(221)
    plt.scatter(Y[:, 0], Y[:, 1], 4)
    plt.axis('equal')
    plt.title("Covering Radius = %.3g, Thresh = %.3g"%(scale, thresh))
    plt.subplot(222)
    plotDGM(Is[0], color = 'b')
    plotDGM(Is[1], color = 'r')
    plotDGM(Is[2], color = 'g')
    plt.xlim([-0.1, 1.5])
    plt.ylim([-0.1, 1.5])
    plt.title("Alpha Complex Persistences")
    plt.subplot(223)
    plt.imshow(res1['PI'], extent = (res1['xr'][0], res1['xr'][-1], \
        res1['yr'][-1], res1['yr'][0]), cmap = 'afmhot', interpolation = 'nearest')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("H1 Persistence Image")
    plt.subplot(224)
    plt.imshow(res2['PI'], extent = (res2['xr'][0], res2['xr'][-1], \
        res2['yr'][-1], res2['yr'][0]), cmap = 'afmhot', interpolation = 'nearest')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("H2 Persistence Image")
    s = filename.split("qrtosso")[0]
    s = s + getPlotName(filename, ".stl") + ".png"
    plt.savefig(s, bbox_inches = 'tight')


if __name__ == '__main__':
    files = glob.glob("BoneData/FIGURE_PORTION/*/*/*.mat")
    #comparePersistenceImages(files, useH2 = True)
    ridgeRegressionPIs(files)
    ridgeRegressionPIs(files, useH1 = False)
    ridgeRegressionPIs(files, useH2 = False)

if __name__ == '__main__2':
    files = glob.glob("BoneData/FIGURE_PORTION/*/*/*.stl")
    plt.figure(figsize=(20, 20))
    for f in files:
        getBonePDs(f)
