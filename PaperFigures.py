import numpy as np
import matplotlib.pyplot as plt

from scipy import spatial
from scipy.ndimage.morphology import distance_transform_edt as edt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sklearn.metrics import pairwise_distances

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import PersistenceImages.persistence_images as pimgs
from HyperoptUtils import *
from BoneData import *
from persim import plot_diagrams
import cechmate as cm


"""
##############################################

            ALPHA FILTRATION FIGURE

##############################################
"""

def drawAlpha(X, filtration, r, draw_balls = False):
    """
    Draw the delaunay triangulation in dotted lines, with the alpha faces at
    a particular scale
    Parameters
    ----------
    X: ndarray(N, 2)
        A 2D point cloud
    filtration: list of [(idxs, d)]
        List of simplices in the filtration, listed by idxs, which indexes into
        X, and with an associated scale d at which the simplex enters the filtration
    r: int
        The radius/scale up to which to plot balls/simplices
    draw_balls: boolean
        Whether to draw the balls (discs intersected with voronoi regions)
    """
    
    # Determine limits of plot
    pad = 0.3
    xlims = [np.min(X[:, 0]), np.max(X[:, 0])]
    xr = xlims[1]-xlims[0]
    ylims = [np.min(X[:, 1]), np.max(X[:, 1])]
    yr = ylims[1]-ylims[0]
    xlims[0] -= xr*pad
    xlims[1] += xr*pad
    ylims[0] -= yr*pad
    ylims[1] += yr*pad

    if draw_balls:
        resol = 2000
        xr = np.linspace(xlims[0], xlims[1], resol)
        yr = np.linspace(ylims[0], ylims[1], resol)
        xpix, ypix = np.meshgrid(xr, yr)
        P = np.ones((xpix.shape[0], xpix.shape[1], 4))
        PComponent = np.ones_like(xpix)
        PBound = np.zeros_like(PComponent)
        # First make balls
        XPix = np.array([xpix.flatten(), ypix.flatten()]).T
        D = pairwise_distances(X, XPix)
        for i in range(X.shape[0]):
            # First make the ball part
            ballPart = (xpix-X[i, 0])**2 + (ypix-X[i, 1])**2 <= r**2
            # Now make the Voronoi part
            voronoiPart = np.reshape(np.argmin(D, axis=0) == i, ballPart.shape)
            Pi = ballPart*voronoiPart
            PComponent[Pi == 1] = 0
            # Make the boundary stroke part
            e = edt(1-Pi)
            e[e > 10] = 0
            e[e > 0] = 1.0/e[e > 0]
            PBound = np.maximum(e, PBound)
        # Now make Voronoi regions
        P[:, :, 0] = PComponent
        P[:, :, 1] = PComponent
        P[:, :, 3] = 0.2 + 0.8*PBound
        plt.imshow(np.flipud(P), cmap='magma', extent=(xlims[0], xlims[1], ylims[0], ylims[1]))

    # Plot simplices
    patches = []
    for (idxs, d) in filtration:
        if len(idxs) == 2:
            if d < r:
                plt.plot(X[idxs, 0], X[idxs, 1], 'k', 2)
            else:
                plt.plot(X[idxs, 0], X[idxs, 1], 'gray', linestyle='--', linewidth=1)
        elif len(idxs) == 3 and d < r:
            patches.append(Polygon(X[idxs, :]))
    ax = plt.gca()
    p = PatchCollection(patches, alpha=0.2, facecolors='C1')
    ax.add_collection(p)
    plt.scatter(X[:, 0], X[:, 1], zorder=0)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    #plt.axis('equal')


def alphaFigure():
    np.random.seed(0)
    X = np.random.randn(20, 2)
    X /= np.sqrt(np.sum(X**2, 1))[:, None]
    X += 0.2*np.random.randn(X.shape[0], 2)


    alpha = cm.Alpha()
    filtration = alpha.build(X)
    dgmsalpha = alpha.diagrams(filtration)

    plt.figure(figsize=(16, 4))
    scales = [0.2, 0.45, 0.9]
    N = len(scales) + 1
    for i, s in enumerate(scales):
        plt.subplot(1, N, i+1)
        if i == 0:
            drawAlpha(X, filtration, s, True)
        else:
            drawAlpha(X, filtration, s, True)
        plt.title("$\\alpha = %.3g$"%s)
    plt.subplot(1, N, N)
    plot_diagrams(dgmsalpha)
    for scale in scales:
        plt.plot([-0.01, scale], [scale, scale], 'gray', linestyle='--', linewidth=1, zorder=0)
        plt.plot([scale, scale], [scale, 1.0], 'gray', linestyle='--', linewidth=1, zorder=0)
        plt.text(scale+0.01, scale-0.01, "%.3g"%scale)
    plt.title("Persistence Diagram")
    plt.savefig("Alpha.svg", bbox_inches='tight')





"""
##############################################

        Bone Ridge Coefficients Figure

##############################################
"""

def get_bone_PI_results(birth_range, pers_range, max_death, pixel_size, sigma, alphas):
    bone_df = get_bone_data_df()
    dgm_df = bone_df[['dgm']]
    y = bone_df['trabnum'].values
    y = y.flatten()

    weight_params = {'n':1}
    kernel_params = {'sigma':sigma}

    fn = lambda dgm: vec_dgm_by_per_images(dgm, birth_range=birth_range, pers_range=pers_range, max_death=max_death, pixel_size=pixel_size, weight=pimgs.weighting_fxns.persistence, weight_params=weight_params, kernel_params=kernel_params)
    X = gen_dgm_feature_df(dgm_df, fn).values

    # Figure out shape and pixel locations by calling once as an example
    imgr = pimgs.PersistenceImager(birth_range=birth_range, pers_range=pers_range, pixel_size=pixel_size, weight=pimgs.weighting_fxns.persistence, weight_params=weight_params, kernel_params=kernel_params)
    img_example = imgr.transform(np.array([[0, 0]]), skew=True)
    img_example = img_example.T

    ## Step 2: Do cross-validation over different alphas for ridge regression
    scorer = make_scorer(mean_squared_error)
    pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(fit_intercept=True))])
    gr = GridSearchCV(pipeline_ridge, cv=6, param_grid={"ridge__alpha":alphas}, scoring=scorer)
    gr.fit(X, y)
    errs = np.sqrt(gr.cv_results_['mean_test_score'])
    return {'errs':errs, 'img_example':img_example, 'X':X, 'y':y}

def get_coeffs(res, alphas, alpha_idx):
    img_example = res['img_example']
    pipeline_ridge = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(fit_intercept=True, alpha=alphas[alpha_idx]))])
    pipeline_ridge.fit(res['X'], res['y'])
    y_est = pipeline_ridge.predict(res['X'])
    coeff = pipeline_ridge.steps[1][1].coef_
    h1coeff = np.reshape(coeff[0:img_example.size], img_example.shape)
    h2coeff = np.reshape(coeff[img_example.size::], img_example.shape)
    return (h1coeff, h2coeff)

def plot_coeffs(h1coeff, h2coeff, birth_range, pers_range, k, title):
    plt.subplot(2, 4, k+2)
    lim = np.max(np.abs(h1coeff))
    plt.imshow(h1coeff, vmin=-lim, vmax=lim, extent = (birth_range[0], birth_range[1], pers_range[1], pers_range[0]), cmap = 'RdBu', interpolation = 'nearest')
    plt.gca().invert_yaxis()
    plt.xlabel("Birth")
    plt.ylabel("Persistence")
    plt.title("%s H1"%title)

    plt.subplot(2, 4, 4+k+2)
    lim = np.max(np.abs(h2coeff))
    plt.imshow(h2coeff, vmin=-lim, vmax=lim, extent = (birth_range[0], birth_range[1], pers_range[1], pers_range[0]), cmap = 'RdBu', interpolation = 'nearest')
    plt.gca().invert_yaxis()
    plt.xlabel("Birth")
    plt.ylabel("Persistence")
    plt.title("%s H2"%title)

def test_alpha_cv_PI_example():
    """
    Choose a set of reasonable parameters and test alpha cross
    validation on the standard scaler + ridge regression pipeline
    on concatenated persistence images from H1 and H2
    """
    alphas = np.logspace(-3, 4, 100)
    birth_range = (0, 0.5)
    pers_range = (0, 0.61)
    max_death = 0.7

    resol_coarse = 0.02
    resol_fine = 0.005
    sigma = 0.05
    res_coarse = get_bone_PI_results(birth_range, pers_range, max_death, pixel_size=resol_coarse, sigma=sigma, alphas=alphas)
    res_fine = get_bone_PI_results(birth_range, pers_range, max_death, pixel_size=resol_fine, sigma=sigma, alphas=alphas)

    ## Plot the results for different alphas for different parameters
    resol = 3
    plt.figure(figsize=(resol*4, resol*2))
    # Plot coarse errors first
    idx_coarse = np.argmin(res_coarse['errs'])
    idx_fine = np.argmin(res_fine['errs'])
    h1coeff, h2coeff = get_coeffs(res_coarse, alphas, idx_coarse)
    plot_coeffs(h1coeff, h2coeff, birth_range, pers_range, 0, "res=$%.3g$, $\\beta=%.3g$"%(resol_coarse, alphas[idx_coarse]))
    h1coeff, h2coeff = get_coeffs(res_fine, alphas, idx_coarse)
    plot_coeffs(h1coeff, h2coeff, birth_range, pers_range, 2, "res=$%.3g$, $\\beta=%.3g$"%(resol_fine, alphas[idx_coarse]))
    h1coeff, h2coeff = get_coeffs(res_fine, alphas, idx_fine)
    plot_coeffs(h1coeff, h2coeff, birth_range, pers_range, 1, "res=$%.3g$, $\\beta=%.3g$"%(resol_fine, alphas[idx_fine]))


    plt.subplot(241)
    plt.plot(alphas, res_coarse['errs'])
    plt.plot(alphas, res_fine['errs'], linestyle='--')
    plt.legend(["Coarse", "Fine"])
    plt.scatter([alphas[idx_coarse]], [res_coarse['errs'][idx_coarse]], c='C0')
    plt.scatter([alphas[idx_coarse]], [res_fine['errs'][idx_coarse]], c='C1')
    plt.scatter([alphas[idx_fine]], [res_fine['errs'][idx_fine]], c='C1')
    plt.gca().set_xscale("log")
    plt.xlabel("$\\beta$")
    plt.ylabel("Trab.Num RMSE (1/mm)")

    plt.tight_layout()
    plt.savefig("RidgeCoeffs.svg", bbox_inches='tight')


if __name__ == '__main__':
    #alphaFigure()
    test_alpha_cv_PI_example()