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
import matplotlib.colors as colors
import pandas as pd
from BoneData import get_bone_data_df

# persistence images routines
import PersistenceImages.persistence_images as pimgs
from HyperoptUtils import vec_dgm_by_per
from persim import plot_diagrams

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os.path


def test_alpha_cv_grabandsort_example():
    """
    Choose a set of reasonable parameters and test alpha cross
    validation on the standard scaler + ridge regression pipeline
    on grabbed and sorted vectors from persistence diagrams
    """
    N = 18
    X = []
    y = []
    plims = [0, 0.5, 0, 0.61]
    ## Step 1: Generate all of the persistence images
    plt.figure(figsize=(12, 6))
    start = 31
    npairs = 65-31
    per_only = True
    for i in range(N):
        plt.clf()
        print(i)
        res = sio.loadmat("PDs/%i.mat"%i)
        y.append(res['trabnum'])
        h = np.array(res['H1'])
        x = vec_dgm_by_per(h, start, npairs, per_only=per_only)
        plt.subplot(121)
        plot_diagrams(h, labels=['H1'], lifetime=True)
        plt.title("trabnum = %.3g"%y[-1])
        plt.xlim(plims[0:2])
        plt.ylim(plims[2::])
        plt.subplot(122)
        plt.plot(x)
        plt.savefig("Pers%i.png"%i, bbox_inches='tight')
        X.append(x.flatten())
    X = np.array(X)
    y = np.array(y)
    y = y.flatten()

    ## Step 2: Do cross-validation over different alphas for ridge regression
    alphas = np.logspace(-4, 4, 100)
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
        coef = pipeline_ridge.steps[1][1].coef_
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
        plt.xlabel("Sorted Persistence Index")
        plt.ylabel("Persistence")
        plt.title("Ridge Regression Coefficients")
        plt.plot(coef)
        plt.savefig("%i.png"%idx, bbox_inches='tight')

def do_bone_gridsearch():
    bone_df = get_bone_data_df()
    dgm_df = bone_df[['dgm']]
    y = bone_df['trabnum'].values
    y = y.flatten()

    N = 150
    per_only=False
    alphas = np.logspace(-4, 4, 100)
    scorer = make_scorer(mean_squared_error)

    all_mses = np.inf*np.ones((N, N, alphas.size))
    for start in range(N):
        for end in range(start+1, N):
            npairs=end-start
            X = []
            for i in range(dgm_df.shape[0]):
                h = dgm_df['dgm'][i]['H1']
                x = vec_dgm_by_per(h, start, npairs, per_only=per_only)
                X.append(x.flatten())
            X = np.array(X)
            pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(fit_intercept=True))])
            gr = GridSearchCV(pipeline_ridge, cv=6, param_grid={"ridge__alpha":alphas}, scoring=scorer)
            gr.fit(X, y)
            errs = gr.cv_results_['mean_test_score']
            all_mses[start, end, :] = errs
            idx = np.argmin(all_mses)
            idx = np.unravel_index(idx, (N, N, alphas.size))
            print("(%i, %i) err %.3g, best so far (%i, %i): %.3g"%(start, end, np.min(errs), idx[0], idx[1], np.min(all_mses)))
        sio.savemat("gridsearch.mat", {"all_mses":all_mses})
    mses_opt = np.min(all_mses, 2)
    alphas_opt = alphas[np.argmin(all_mses, 2)]
    plt.figure(figsize=(12, 6))
    J, I = np.meshgrid(np.arange(N)+1, np.arange(N)+1)
    plt.subplot(121)
    pcm = plt.gca().pcolor(J, I, mses_opt.T, cmap='magma_r')
    plt.gcf().colorbar(pcm, ax=plt.gca(), extend='max')
    plt.xlabel("Start Index")
    plt.ylabel("End Index")
    plt.title("MSEs")
    plt.subplot(122)
    pcm = plt.gca().pcolor(J, I, alphas_opt.T, norm=colors.LogNorm(vmin=alphas_opt.min()+1e-5, vmax=alphas_opt.max()), cmap='magma_r')
    plt.gcf().colorbar(pcm, ax=plt.gca(), extend='max')
    plt.xlabel("Start Index")
    plt.ylabel("End Index")
    plt.title("Alphas")
    plt.show()



if __name__ == '__main__':
    test_alpha_cv_grabandsort_example()
    #do_bone_gridsearch()
