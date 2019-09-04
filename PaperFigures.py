import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams
import cechmate as cm
from scipy import spatial
from scipy.ndimage.morphology import distance_transform_edt as edt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sklearn.metrics import pairwise_distances


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

if __name__ == '__main__':
    alphaFigure()