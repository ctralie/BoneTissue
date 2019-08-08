"""
Programmer: Chris Tralie
Purpose: To compute alpha filtrations for bone data.
Requires meshlab to be installed for loading the mesh files
"""
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import glob
import pandas
import json
import subprocess
import cechmate as cm
from persim import plot_diagrams as plot_dgms
from PolyMesh import *

def getPlotName(f, ext):
    s = f.split("qrtosso_")[1]
    s1 = s[0:2]
    s2 = s.split("rec_conv_")[1]
    s2 = s2.split(ext)[0]
    s = "%s_%s"%(s1, s2)
    return s

def getBoneAlpha(filename, hull_samples = 0):
    """
    Compute the Alpha filtration of a bone specimen
    Parameters
    ----------
    filename: string
        Path to 3D mesh file containing the bone
    hull_samples: int
        The number of points to sample on the convex hull.
        This can help increase the birth time of H2
    Returns
    -------
    {'X': ndarray(N, 3)
        The original point cloud,
     'Y': ndarray(N+hull_samples, 3)
        The point cloud with hull points added,
     'dgms': [ndarray(N0, 2), ndarray(N1, 2), ndarray(N2, 2)]
        A list of diagrams for H0, H1, and H2
    }
    """
    ## Step 1: Load mesh by using meshlab to convert it to off
    ## In the process, remove duplicate vertices
    filetemp = "%s.off"%filename[0:-4]
    subprocess.call(["meshlabserver", "-i", filename, "-s", "removedup.mlx", "-o", filetemp])
    (X, _, _) = loadOffFile(filetemp)
    # Add some noise to put into general position (since points were sampled
    # on a grid, bad things can happen otherwise)
    X = X + 0.001*np.random.randn(X.shape[0], X.shape[1])
    
    ## Step 2: Sample convex hull, if requested
    Y = np.array(X)
    if hull_samples > 0:
        hull = ConvexHull(X)
        Ps, _ = randomlySamplePoints(X, hull.simplices, hull_samples)
        saveOffFile("hull%i.off"%hull_samples, Ps, np.array([]), np.array([]))
        Y = np.concatenate((Y, Ps))
        saveOffFile("hull%i_withmesh.off"%hull_samples, Y, np.array([]), np.array([]))

    ## Step 3: Do the Alpha filtration
    alpha = cm.Alpha()
    filtration = alpha.build(Y)
    dgms = alpha.diagrams(filtration)

    return {'X':X, 'Y':Y, 'dgms':dgms}

def computeAllPersistenceDiagrams(hull_samples=0):
    """
    Compute all alpha filtrations for the bone dataset, and save
    the results along with the parameters in the dataset
    """
    ## Step 1: Load in parameters
    TrabNums = pandas.read_csv("TrabNums.csv")
    specimens = [s.lower() for s in TrabNums['Specimen'].values.tolist()]
    translate = {'Trab num. (1/mm)':'trabnum', 'Trab tick. (mm)':'trabtick', 'Trab. lgth. (mm)':'trablen', 'BV/TV (%)':'bv_tv'}
    specimen2val = {specimens[i]:{translate[s]:TrabNums[s].values[i] for s in translate} for i in range(len(specimens))}

    plt.figure(figsize=(16, 8))
    for i, f in enumerate(glob.glob("BoneData/FIGURE_PORTION/*/*/*.stl")):
        print(f)
        res = getBoneAlpha(f, hull_samples=hull_samples)
        dgms, Y = res['dgms'], res['Y']
        specimen = f.split("qrtosso_")[1][0:2]
        res = copy.deepcopy(specimen2val[specimen])
        plt.clf()
        plt.subplot(121)
        plt.scatter(Y[:, 0], Y[:, 1], 1)
        plt.axis('equal')
        plt.title("XY Slice, Specimen %s\n%s"%(specimen, json.dumps({s:"%.3g"%res[s] for s in res})))
        plt.subplot(122)
        plot_dgms(dgms)
        plt.title("Alpha Persistence Diagrams")
        plt.savefig("PDs/%i_%i.png"%(i, hull_samples), bbox_inches='tight')
        for k, dgm in enumerate(dgms):
            res['H%i'%k] = dgm
        res['specimen'] = specimen
        sio.savemat("PDs/%i_%i.mat"%(i, hull_samples), res)

        
if __name__ == '__main__':
    computeAllPersistenceDiagrams(hull_samples=10000)