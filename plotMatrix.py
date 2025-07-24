import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

def plotMatrix(mat, xvec, yvec, xlab='', ylab='', stretch=False, planet_posi=None, scale='linear', power=2):
    """
    Plots a 2D spectra with given x and y vectors, with options for stretching, scaling, and marking a planet position.

    Parameters:
    mat (2D array): The matrix to be plotted.
    xvec (1D array): The x-axis values.
    yvec (1D array): The y-axis values.
    xlab (str): Label for the x-axis. Default is an empty string.
    ylab (str): Label for the y-axis. Default is an empty string.
    stretch (bool): If True, stretches the color scale to mean ± 8*std. Default is False.
    planet_posi (float or None): Position of the planet to be marked with a horizontal line. Default is None.
    scale (str): Scale of the color map, either 'linear' or 'log'. Default is 'linear'.
    power (int): Power for the logarithmic scale. Default is 2.
    """
    if stretch:
        vmin = np.nanmean(mat) - np.nanstd(mat) * 8
        vmax = np.nanmean(mat) + np.nanstd(mat) * 8
    else:
        vmin = np.nanmin(mat)
        vmax = np.nanmax(mat)
    ext = [xvec[0], xvec[-1], yvec[0], yvec[-1]]

    if scale == 'linear' or scale is None:
        plt.figure(figsize=(12, 3))
        plt.imshow(mat, origin='lower', aspect='auto', extent=ext, vmin=vmin, vmax=vmax)
        if planet_posi is not None:
            plt.hlines(y=planet_posi, xmin=xvec[0], xmax=xvec[-1], ls='--', colors='lightgray', alpha=1, linewidth=2)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.colorbar()

    elif scale == 'log':
        plt.figure(figsize=(12, 3))
        Norm = matplotlib.colors.SymLogNorm(linthresh=1, linscale=1, vmin=vmin, vmax=vmax, base=10**power)
        plt.imshow(mat, origin='lower', aspect='auto', extent=ext, norm=Norm)
        
        if planet_posi is not None:
            plt.hlines(y=planet_posi, xmin=xvec[0], xmax=xvec[-1], ls='--', colors='lightgray', alpha=1, linewidth=2)

        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.colorbar()

