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
from PolyMesh import loadOffFileExternal

def getPlotName(f, ext):
    s = f.split("qrtosso_")[1]
    s1 = s[0:2]
    s2 = s.split("rec_conv_")[1]
    s2 = s2.split(ext)[0]
    s = "%s_%s"%(s1, s2)
    return s

def getBoneAlpha(filename, sample_hull = False):
    """
    Compute the Alpha filtration of a bone specimen
    Parameters
    ----------
    filename: string
        Path to 3D mesh file containing the bone
    sample_hull: boolean
        Whether to sample points on the convex hull to help
        increase the birth time of H2
    Returns
    -------
    Y: ndarray(N, 3)
        The original point cloud
    dgms: [ndarray(N0, 2), ndarray(N1, 2), ndarray(N2, 2)]
        A list of diagrams for H0, H1, and H2
    """
    ## Step 1: Load mesh by using meshlab to convert it to off
    ## In the process, remove duplicate vertices
    filetemp = "%s.off"%filename[0:-4]
    subprocess.call(["meshlabserver", "-i", filename, "-s", "removedup.mlx", "-o", filetemp])
    (Y, _, _) = loadOffFileExternal(filetemp)
    Y = Y + 0.001*np.random.randn(Y.shape[0], Y.shape[1])
    
    ## Step 2: Sample convex hull, if requested
    ##TODO: FINISH THIS

    ## Step 3: Do the Alpha filtration
    alpha = cm.Alpha()
    filtration = alpha.build(Y)
    dgms = alpha.diagrams(filtration)

    return Y, dgms

def computeAllPersistenceDiagrams():
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
        Y, dgms = getBoneAlpha(f)
        specimen = f.split("qrtosso_")[1][0:2]
        res = copy.deepcopy(specimen2val[specimen])
        plt.clf()
        plt.subplot(121)
        plt.scatter(Y[:, 0], Y[:, 1], 1)
        plt.axis('equal')
        plt.title("XY Slice, Specimen %s\n%s"%(specimen, json.dumps({s:"%.3g"%res[s] for s in res})))
        plt.subplot(122)
        #plot_dgms(dgms)
        plt.title("Alpha Persistence Diagrams")
        plt.savefig("PDs/%i.png"%i, bbox_inches='tight')
        for k, dgm in enumerate(dgms):
            res['H%i'%k] = dgm
        res['specimen'] = 'specimen'
        sio.savemat("PDs/%i.mat"%i, res)

        
if __name__ == '__main__':
    computeAllPersistenceDiagrams()