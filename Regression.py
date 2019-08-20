"""
Programmer: Francis Motta, edited by Chris Tralie
"""
import csv
import time
import pickle
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

# persistence images routines
import PersistenceImages.persistence_images as pimgs

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os.path


def get_bone_data_df():
    """
    Return the bone data as data frames
    """
    N = 18
    bone_data = [sio.loadmat("PDs/%i.mat"%i) for i in range(N)]
    vals = ['trabnum', 'trabtick', 'trablen', 'bv_tv']
    vals = {v:[bone_data[i][v].flatten()[0] for i in range(N)] for v in vals}
    vals['dgm'] = [{h:bone_data[i][h] for h in ['H1', 'H2']} for i in range(N)]
    return pd.DataFrame(vals)

def test_alpha_cv_example():
    """
    Choose a set of reasonable parameters and test alpha cross
    validation on the standard scaler + ridge regression pipeline
    """
    N = 18
    X = []
    y = []
    plims = [0, 1, 0, 1]
    shape = ()
    ## Step 1: Generate all of the persistence images
    for i in range(N):
        print(i)
        res = sio.loadmat("PDs/%i.mat"%i)
        y.append(res['trabnum'])
        imgr = pimgs.PersistenceImager(birth_range=(0, 1), 
                                    pers_range= (0, 1), 
                                    pixel_size=0.025,
                                    weight=pimgs.weighting_fxns.persistence, 
                                    weight_params={'n':1}, 
                                    kernel=pimgs.kernels.bvncdf, 
                                    kernel_params={'sigma':0.1})
        x = imgr.transform(res['H1'], skew=True)
        shape = x.shape
        X.append(x.flatten())
    X = np.array(X)
    y = np.array(y)
    y = y.flatten()

    ## Step 2: Do cross-validation over different alphas for ridge regression
    alphas = np.logspace(-3, 2, 100)
    scorer = make_scorer(mean_squared_error)
    pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(fit_intercept=True))])
    gr = GridSearchCV(pipeline_ridge, cv=6, param_grid={"ridge__alpha":alphas}, scoring=scorer)
    gr.fit(X, y)
    errs = gr.cv_results_['mean_test_score']

    ## Step 3: Plot the results for different alphas, applied to the whole dataset
    plt.figure(figsize=(18, 6))
    for idx in range(errs.size):
        pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(fit_intercept=True, alpha=alphas[idx]))])
        pipeline_ridge.fit(X, y)
        y_est = pipeline_ridge.predict(X)
        img = pipeline_ridge.steps[1][1].coef_
        img = np.reshape(img, shape)
        plt.clf()
        plt.subplot(131)
        plt.plot(alphas, errs)
        plt.scatter([alphas[idx]], [errs[idx]])
        plt.gca().set_xscale("log")
        plt.xlabel("$\\alpha$")
        plt.ylabel("MSE")
        plt.title("Bone Data Ridge Regression Test")
        plt.subplot(132)
        plt.scatter(y, y_est)
        plt.title("$\\alpha = %.3g$, MSE = %.3g"%(alphas[idx], errs[idx]))
        plt.xlabel("True Trabecular Number")
        plt.ylabel("Predicted Trabecular Number")
        plt.subplot(133)
        lim = np.max(np.abs(img))
        plt.imshow(img, vmin=-lim, vmax=lim, extent = (plims[0], plims[1], plims[3], plims[2]), cmap = 'RdBu', interpolation = 'nearest')
        plt.gca().invert_yaxis()
        plt.xlabel("Birth")
        plt.ylabel("Persistence")
        plt.savefig("%i.png"%idx, bbox_inches='tight')

def do_bone_gridsearch():
    #TODO: FINISH THIS
    bone_df = get_bone_data_df()
    dgm_df = bone_df[['dgm']]
    y = bone_df['trabnum'].values
    scorer = make_scorer(mean_squared_error)
    
    # setup the imager object with specified parameters
    imgr = pimgs.PersistenceImager(birth_range=(0, 1), 
                                   pers_range= (0, 1), 
                                   pixel_size=pixel_size,
                                   weight=pimgs.weighting_fxns.persistence, 
                                   weight_params=weight_params, 
                                   kernel=pimgs.kernels.bvncdf, 
                                   kernel_params=kernel_params)
    
    pixels_sizes = np.linspace(0.01, 0.2, 10)
    sigmas = np.linspace(0.01, 0.4, 10)
    weight_params = np.linspace(1.0, 3.0, 10)

    pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
    #TODO: FINISH THIS    

if __name__ == '__main__':
    test_alpha_cv_example()