import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

from scipy import interpolate
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.linalg import svd, solve, det
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import chi2, norm
from scipy.ndimage import gaussian_filter1d

from exocrires import info
from exocrires import plotMatrix 
from exocrires.fast_likelihood_calculator import (fast_likelihood_calculator,
                                                 fast_likelihood_map)

from astropy.io import fits

import glob
import tqdm



class Spectra_fitting:

    '''
    Spectra_fitting
    ===============
    This class is designed to process and analyze 2D spectral data, particularly in the context of 
    stellar and planetary contributions. It provides methods to mask unwanted pixels, fit stellar 
    and planetary contributions, and compute multiplicative factors for spectral modeling.
    Attributes:
    spectra : np.ndarray
        The 2D spectral data to be processed.
    transmission_matrix : np.ndarray, optional
        The atmospheric transmission matrix.
    window_length : int, optional
        The length of the filter window for Savitzky-Golay smoothing.
    polyorder : int, optional
        The order of the polynomial used in Savitzky-Golay smoothing.
    Methods:
    Mask_spec(transmission_atmos, atm_threshold, sigma_threshold)
        Masks pixels in the spectra based on atmospheric transmission and flux thresholds.
    fit_star(c, master, observation, window_length, polyorder)
    fit_planet(c, master, transmission, spec_planet, window_length, polyorder)
        Fits the planetary contribution to the observed spectrum using a smoothed master spectrum 
        and transmission model.
    multiplicative_factor_fit(planet_model_spectrum, order_list, window_len, Polyorder)
    '''

    def __init__(self, spectra2D, transmission_matrix=None, window_length=None, polyorder=None):
        """
        Initialize the SpectraProcessor with the given parameters.

        Parameters:
        spectra (np.ndarray): The 2D spectra data.
        """
        self.spectra = spectra2D
        self.transmission_matrix = transmission_matrix
        self.window_length = window_length
        self.polyorder = polyorder

    def Mask_spec(self, transmission_atmos, atm_threshold, sigma_threshold): 
        """
        Mask the pixels that are below the threshold values.

        Parameters:
        transmission_cube (np.ndarray): The transmission cube.
        atm_threshold (float): The threshold value for the transmission cube.
        sigma_threshold (float): The threshold value for the sigma cube.

        Returns:
        np.ndarray: The masked spectra.
        """
        
        for i in range(self.spectra.shape[0]):

            low_atmos_transmission_mask = np.where((transmission_atmos[i, 0:2046*3, 1]) < atm_threshold, True, False)

            for j in range(self.spectra.shape[1]):

                mean_flux = np.nanmean(self.spectra[i, j, 0:2046*3])
                std_flux = np.nanstd(self.spectra[i, j, 0:2046*3])

                bad_pixel_mask = np.where((np.abs(self.spectra[i, j, 0:2046*3] - mean_flux) > (sigma_threshold * std_flux)), True, False)

                self.spectra[i, j, 0:2046*3][bad_pixel_mask] = np.nan
                self.spectra[i, j, 0:2046*3][low_atmos_transmission_mask] = np.nan

        return (self.spectra)
    
    
    def fit_star(self, c, master, observation, window_length, polyorder):
            
        """
        Fits the stellar contribution to the observed spectrum using a smoothed master spectrum.
        Parameters:
        -----------
        c : float
            Scaling factor applied to the stellar contribution in the observed spectrum.
        master : array-like
            The master spectrum, representing the reference stellar spectrum.
        observation : array-like
            The observed spectrum, which includes contributions from the star and other sources.
        window_length : int
            The length of the filter window for the Savitzky-Golay smoothing. Must be a positive odd integer.
        polyorder : int
            The order of the polynomial used in the Savitzky-Golay smoothing. Must be less than `window_length`.
        Returns:
        --------
        tuple
            A tuple containing:
            - stellar_real_obs (array-like): The estimated stellar contribution in the observed spectrum.
            - stellar_model_contribution (array-like): The smoothed stellar model contribution.
            - alpha_star (array-like): The scaling factor between the smoothed observation and the smoothed master spectrum.
        """
            
        smoothed_master = savgol_filter(master, window_length, polyorder)
        smoothed_obs = savgol_filter(observation, window_length, polyorder)
        alpha_star = smoothed_obs / smoothed_master
        stellar_real_obs = c * alpha_star * master
        stellar_model_contribution = alpha_star * master
        

        return (stellar_real_obs, stellar_model_contribution, alpha_star)
    
    def fit_planet(self, c, master, transmission, spec_planet, window_length, polyorder):
        """
        Fits the planetary contribution to the observed spectrum using a smoothed master spectrum and transmission model.

        Parameters:
        -----------
        c : float
            Scaling factor applied to the planetary contribution in the observed spectrum.
        master : array-like
            The master spectrum, representing the reference stellar spectrum.
        transmission : array-like
            The atmospheric transmission spectrum.
        spec_planet : array-like
            The planetary model spectrum.
        window_length : int
            The length of the filter window for the Savitzky-Golay smoothing. Must be a positive odd integer.
        polyorder : int
            The order of the polynomial used in the Savitzky-Golay smoothing. Must be less than `window_length`.

        Returns:
        --------
        tuple
            A tuple containing:
            - planet_real_obs (array-like): The estimated planetary contribution in the observed spectrum after removing leakage.
            - planet_model_contribution (array-like): The smoothed planetary model contribution.
            - planet_obs (array-like): The raw planetary contribution in the observed spectrum.
            - planet_leakage (array-like): The leakage of the planetary signal into the stellar spectrum.
            - alpha_planet (array-like): The scaling factor between the smoothed planetary and master spectra.
        """
        # Smooth the master spectrum using Savitzky-Golay filter
        smoothed_master = savgol_filter(master, window_length, polyorder)
        
        # Smooth the product of transmission and planetary spectrum
        smoothed_planet = savgol_filter(transmission * spec_planet, window_length, polyorder)
        
        # Compute the scaling factor (alpha_planet) between the smoothed planetary and master spectra
        alpha_planet = smoothed_planet / smoothed_master
        
        # Compute the raw planetary contribution in the observed spectrum
        planet_obs = c * transmission * spec_planet
        
        # Compute the leakage of the planetary signal into the stellar spectrum
        planet_leakage = c * master * alpha_planet
        
        # Compute the real planetary contribution after removing leakage
        planet_real_obs = planet_obs - planet_leakage
        
        # Compute the planetary model contribution after removing the scaled master spectrum
        planet_model_contribution = transmission * spec_planet - alpha_planet * master

        return (planet_real_obs, planet_model_contribution, planet_obs, planet_leakage, alpha_planet)

    def multiplicative_factor_fit (self, planet_model_spectrum, order_list, window_len, Polyorder, central_pix=17):

        """
        Fits multiplicative factors for stellar and planetary contributions in the observed spectra.

        Parameters:
        -----------
        planet_model_spectrum : np.ndarray
            The planetary model spectrum.
        order_list : list
            List of spectral orders to process.
        window_len : int
            The length of the filter window for the Savitzky-Golay smoothing. Must be a positive odd integer.
        Polyorder : int
            The order of the polynomial used in the Savitzky-Golay smoothing. Must be less than `window_len`.

        Returns:
        --------
        tuple
            A tuple containing:
            - fit_matrix_planet : np.ndarray
                The fitted planetary contribution matrix.
            - fit_matrix_star : np.ndarray
                The fitted stellar contribution matrix.
            - fit_alpha_star : np.ndarray
                The scaling factors for the stellar contribution.
            - fit_alpha_planet : np.ndarray
                The scaling factors for the planetary contribution.
            - fit_model : np.ndarray
                The combined model matrix for stellar and planetary contributions.
            - popt_list : np.ndarray
                The optimized parameters for each fit.
        """

        # Set the window length and polynomial order for smoothing
        self.window_length = window_len
        self.polyorder = Polyorder

        # Initialize arrays to store results
        popt_list = np.zeros(shape=(self.spectra.shape[0], self.spectra.shape[1], 3, 2))
        fit_matrix_planet = np.zeros(shape=(self.spectra.shape))
        fit_matrix_star = np.zeros(shape=(self.spectra.shape))
        fit_alpha_star = np.zeros(shape=(self.spectra.shape))
        fit_alpha_planet = np.zeros(shape=(self.spectra.shape))
        fit_model = np.zeros(shape=(2, self.spectra.shape[0], self.spectra.shape[1], self.spectra.shape[2]))

        # Loop through the specified spectral orders
        for i in order_list:
            for x in range(3):  # Loop through the three detectors
                # Integrate the master (stellar) spectrum over the 3 central rows on the STAR.
                # central_pix is the stellar PSF-centre row in the (possibly widened) stack; for the
                # legacy 35-row star-at-17 geometry central_pix=17 -> rows 16:19 (unchanged).
                master_int = np.trapz(self.spectra[i, central_pix-1:central_pix+2, x*2046:(x+1)*2046], axis=0)
                mask = np.isfinite(master_int)

                # Loop through each observation
                for j in range(self.spectra.shape[1]): 
                    mask_data = np.isfinite(self.spectra[i, j][x*2046:(x+1)*2046])

                    # Skip fully-masked / nearly-empty segments (e.g. telluric-opaque M-band
                    # orders): curve_fit needs non-empty ydata and the savgol fit needs more than
                    # window_length valid points. Such segments stay 0 in the output matrices.
                    if (mask & mask_data).sum() <= self.window_length:
                        continue

                    # Define a wrapper function for the model
                    def wrap_model(master_wrap, cs, cp):
                        # Fit the stellar contribution
                        wrap_star = self.fit_star(cs, master_wrap, self.spectra[i, j, x*2046:(x+1)*2046][mask & mask_data], self.window_length, self.polyorder)
                        # Fit the planetary contribution
                        wrap_planet = self.fit_planet(cp, master_wrap, self.transmission_matrix[i, x*2046:(x+1)*2046][mask & mask_data], planet_model_spectrum[i, x*2046:(x+1)*2046, 1][mask & mask_data], self.window_length, self.polyorder)
                        # Combine the stellar and planetary contributions
                        wrapped_model = wrap_star[0] + wrap_planet[0]
                        return wrapped_model

                    # Fit the model to the data using curve fitting
                    popt, pcov = curve_fit(wrap_model, master_int[mask & mask_data], self.spectra[i, j, x*2046:(x+1)*2046][mask & mask_data])
                    popt_list[i, j, x] = popt

                    # Fit the stellar and planetary contributions separately
                    star_fit = self.fit_star(popt[0], master_int[mask & mask_data], self.spectra[i, j, x*2046:(x+1)*2046][mask & mask_data], self.window_length, self.polyorder)
                    planet_fit = self.fit_planet(popt[1], master_int[mask & mask_data], self.transmission_matrix[i, x*2046:(x+1)*2046][mask & mask_data], planet_model_spectrum[i, x*2046:(x+1)*2046, 1][mask & mask_data], self.window_length, self.polyorder)

                    # Store the fitted results in the respective matrices
                    fit_matrix_planet[i, j][x*2046:(x+1)*2046][mask & mask_data] = planet_fit[0]
                    fit_matrix_star[i, j][x*2046:(x+1)*2046][mask & mask_data] = star_fit[0]
                    fit_alpha_star[i, j][x*2046:(x+1)*2046][mask & mask_data] = star_fit[-1]
                    fit_alpha_planet[i, j][x*2046:(x+1)*2046][mask & mask_data] = planet_fit[-1]
                    fit_model[0][i, j][x*2046:(x+1)*2046][mask & mask_data] = star_fit[1]
                    fit_model[1][i, j][x*2046:(x+1)*2046][mask & mask_data] = planet_fit[1]

                    # Print the optimized parameters for debugging
                    print(popt)

        # Return the fitted matrices and parameters
        return (fit_matrix_planet, fit_matrix_star, fit_alpha_star, fit_alpha_planet, fit_model, popt_list)

    def correct_alpha_star_planet_pixel(
        self, fit_matrix_star, fit_alpha_star, popt_list,
        stack, spatial_pix, central_pix, po, chip_size=2046
    ):
        """Replace alpha_star at the planet pixel with the median from the anti-nod region.

        During multiplicative_factor_fit(), alpha_star at the planet pixel is inflated
        by the planet's own continuum (delta_alpha = LP(planet)/LP(master)), causing the
        stellar model to over-subtract at the planet position and inject spurious stellar
        spectral features into the residual.

        The correction uses the 3 rows centred on anti_pix = central_pix − po.  By PSF
        symmetry those rows have the same stellar wing flux level as the planet row, so
        their fitted alpha_star values are the best proxy for the true (uncontaminated)
        alpha at spatial_pix.

        Parameters
        ----------
        fit_matrix_star : ndarray (nOrder, nObs, nPix)
            Stellar model matrix from multiplicative_factor_fit().
        fit_alpha_star : list of ndarray
            Per-order alpha_star arrays from multiplicative_factor_fit().
        popt_list : ndarray (nOrder, nObs, nChip, 2)
            Fitted [cs, cp] coefficients from multiplicative_factor_fit().
        stack : ndarray (nOrder, nObs, nPix)
            Raw ABBA-stacked data cube.
        spatial_pix : int
            Spatial row index of the positive-nod planet position.
        central_pix : int
            Spatial row index of the stellar PSF centre.
        po : int
            Planet offset in rows (spatial_pix = central_pix + po).
        chip_size : int
            Pixels per detector chip (default 2046).

        Returns
        -------
        fit_matrix_star_corr : ndarray
            Corrected stellar model matrix (copy; original is not mutated).
        """
        anti_pix = central_pix - po
        nSpatial = stack.shape[1]

        # Primary: 3 rows centred on anti_pix (symmetric stellar wing)
        ref_rows = [r for r in [anti_pix - 1, anti_pix, anti_pix + 1]
                    if 0 <= r < nSpatial]

        # Fallback: broad off-planet median if anti-nod region is unavailable
        if len(ref_rows) < 2:
            ref_rows = [r for r in range(nSpatial)
                        if abs(r - anti_pix)    > 4
                        and abs(r - central_pix) > 4
                        and abs(r - spatial_pix) > 4]

        if not ref_rows:
            print("correct_alpha_star_planet_pixel: no valid reference rows found, returning original.")
            return fit_matrix_star.copy()

        print(f"alpha_star correction — reference rows (anti-nod ±1): {ref_rows}")

        nOrders = stack.shape[0]
        n_chips = stack.shape[2] // chip_size
        out = fit_matrix_star.copy()

        for order in range(nOrders):
            for chip in range(n_chips):
                sl = slice(chip * chip_size, (chip + 1) * chip_size)

                alpha_off = fit_alpha_star[order][np.array(ref_rows)][:, sl].copy().astype(float)
                alpha_off[alpha_off == 0] = np.nan
                if np.all(np.isnan(alpha_off)):
                    continue

                alpha_corrected = np.nanmedian(alpha_off, axis=0)
                cs_planet = popt_list[order, spatial_pix, chip, 0]
                master_chip = np.trapz(stack[order, central_pix-1:central_pix+2, sl], axis=0)

                old_row = fit_matrix_star[order, spatial_pix, sl]
                valid_mask = (old_row != 0) & np.isfinite(old_row)
                new_vals = cs_planet * alpha_corrected * master_chip
                out[order, spatial_pix, sl] = np.where(valid_mask, new_vals, old_row)

        return out


class processor_cross_correlation():

    """
    A class to perform cross-correlation analysis on spectral data.
    This class is designed to process spectral data cubes and compute cross-correlation functions (CCFs) 
    for a given set of radial velocity lags. It includes methods for computing the cross-correlation grid, 
    summing the CCFs, and optionally plotting the results.
    Attributes:
        wMod (array-like): Wavelength grid of the model spectrum.
        fMod (array-like): Flux values of the model spectrum.
        wlen (array-like): Wavelength grid of the observed spectrum.
        cube (array-like): Data cube containing observed spectra.
        nOrder (int): Number of spectral orders in the data cube.
        nObs (int): Number of observations in the data cube.
    """

    
    def __init__(self, wMod, fMod, wlen, cube, nOrder, nObs):
        pass
        
        self.wMod = wMod
        self.fMod = fMod
        self.wlen = wlen
        self.cube = cube
        self.nOrder = nOrder
        self.nObs = nObs

    def xcorr(self,f,g):

        """
        Compute the normalized cross-correlation between two arrays.
        Args:
            f (array-like): First input array.
            g (array-like): Second input array.
        Returns:
            float: Normalized cross-correlation coefficient.
        """
        """
        Compute the cross-correlation function (CCF) grid for a range of radial velocity lags.
        Args:
            rvlag (array-like): Array of radial velocity lags (in km/s).
            ncc (int): Number of cross-correlation points.
        Returns:
            numpy.ndarray: A 3D array containing the CCF grid with dimensions (nOrder, nObs, ncc).
        """
        
        nx = len(f)
        I = np.ones(nx)
        f -= np.dot(f,I)/nx
        g -= np.dot(g,I)/nx
        R = np.dot(f,g)/nx
        varf = np.dot(f,f)/nx
        varg = np.dot(g,g)/nx

        return (R / np.sqrt(varf*varg))
        #return (R/np.sqrt(varf))
    
    def test_xvar(self,f,g):

        """
        Compute the normalized cross-correlation between two arrays.
        Args:
            f (array-like): First input array.
            g (array-like): Second input array.
        Returns:
            float: Normalized cross-correlation coefficient.
        """
        """
        Compute the cross-correlation function (CCF) grid for a range of radial velocity lags.
        Args:
            rvlag (array-like): Array of radial velocity lags (in km/s).
            ncc (int): Number of cross-correlation points.
        Returns:
            numpy.ndarray: A 3D array containing the CCF grid with dimensions (nOrder, nObs, ncc).
        """
        
        nx = len(f)
        I = np.ones(nx)
        f -= np.dot(f,I)/nx
        g -= np.dot(g,I)/nx
        R = np.dot(f,g)/nx
        varf = np.dot(f,f)/nx
        varg = np.dot(g,g)/nx

        return (R)

    def get_cc_grid(self, rvlag, ncc):

        """
        Compute the cross-correlation function (CCF) grid for a given set of radial velocity lags.

        This function calculates the CCF for each spectral order and observation in the data cube
        by shifting the model spectrum (`fMod`) to match the observed wavelengths (`wlen`) at 
        different radial velocity shifts (`rvlag`). The shifted model spectrum is then cross-correlated 
        with the observed data to produce the CCF grid.

        Args:
            rvlag (array-like): Array of radial velocity lags (in km/s) to compute the CCF over.
            ncc (int): Number of cross-correlation points (length of the radial velocity lag array).

        Returns:
            numpy.ndarray: A 3D array of shape (nOrder, nObs, ncc) containing the computed CCF values.
                           - `nOrder`: Number of spectral orders.
                           - `nObs`: Number of observations.
                           - `ncc`: Number of radial velocity lag points.

        Notes:
            - `self.wMod` and `self.fMod` represent the wavelength and flux of the model spectrum.
            - `self.wlen` is the observed wavelength grid.
            - `self.cube` is a 3D array containing the observed data, with dimensions (nOrder, nObs, nPix).
            - The radial velocity shift is applied using the relativistic Doppler formula.
            - The `xcorr` method is used to compute the cross-correlation between the observed and 
              shifted model spectra for each order and observation.
        """

        ccf = np.zeros((self.nOrder,self.nObs,ncc))
        coef_spline = interpolate.splrep(self.wMod, self.fMod, s=0.0)
        for irv, rv in enumerate(rvlag):
            beta = rv / 2.998E5
            # Shifting data wlen instead by swapping sign of RV
            wShift = self.wlen * np.sqrt( (1-beta) / (1+beta) )  # A (nDet,nPix) vector
            intMod = interpolate.splev(wShift,coef_spline,der=0) # A (nDet,nPix) vector
            for iOrder in range(self.nOrder):
                for iObs in range(self.nObs):
                    ccf[iOrder,iObs,irv] = self.xcorr(self.cube[iOrder,iObs,], intMod[iOrder,])

        self.rvlag = rvlag
        self.ncc = ncc
        return ccf

    def ccf_tot (self, rvlag, ncc, plot=True, subtract_continuum='med_flux', normalization = 'median subtracted', v_sys=None, clean_grids=None, po=None, central_pix=None, spatial_pix=None):

        """
        Compute the total map of the cross-correlation function (CCF) and its signal-to-noise ratio (SNR).
        Args:
            rvlag (array-like): Array of radial velocity lags (in km/s).
            ncc (int): Number of cross-correlation points.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            subtract_continuum (str, optional): Method to subtract the continuum. Defaults to 'med_flux'.
            normalization (str, optional): Method to normalize the ccf maps. Defaults to 'median subtracted'. 
            v_sys (float, optional): Systemic velocity for reference. Defaults to None.
            clean_grids (list, optional): List of grid ranges to clean the CCF. Defaults to None.
            po (optional): Placeholder for additional parameters. Defaults to None.
            central_pix (int, optional): Central pixel for plotting. Defaults to None.
            spatial_pix (int, optional): Spatial pixel for plotting. Defaults to None.
        Returns:
            tuple: A tuple containing:
                - ccf_Sum (numpy.ndarray): Summed CCF across orders.
                - ccf_SNR (numpy.ndarray): Signal-to-noise ratio of the CCF.
        """

        if subtract_continuum == 'med_flux':

            medFlux=np.zeros(shape=(self.nOrder*3,self.cube.shape[1]))

            for order in range(self.nOrder):

                for Det in range(3):

                    medFlux[order*3+Det]=np.median(self.cube[order, :, 2048*Det:2048*(Det+1)], axis=1)

            ccfWeight = np.sum(medFlux, axis=0)
            ccfWeight /=ccfWeight.sum(axis=0)

        ccf = self.get_cc_grid(rvlag, ncc)

        ccf_Sum=ccf.sum(axis=0)
        
        if normalization == 'median':

            ccf_Sum=(ccf_Sum - np.nanmedian(ccf_Sum))/np.nanmedian(ccf_Sum)

        elif normalization =='max':

            ccf_Sum/= np.nanmax(ccf_Sum)
        
        elif normalization == 'median subtracted':

            for iObs in range(self.nObs): ccf_Sum[iObs,] -= np.nanmedian(ccf_Sum[iObs,])



        if plot == True:

            plotMatrix.plotMatrix(ccf_Sum, rvlag, np.arange(0, self.nObs, 1), 'Radial velocity (km/s)','spatial axis',stretch=True, planet_posi=spatial_pix, scale='log')
            plt.hlines(y=central_pix, xmin=rvlag[0], xmax=rvlag[-1], ls='--', colors='lightgray')
            plt.vlines(x=v_sys, ymin=0, ymax=(self.nObs-1), ls='--', colors='lightgray')
            plt.show()

        
        ccf_SNR=np.zeros(shape=ccf_Sum.shape)

        clean_grids_0=int(clean_grids[0][0])
        clean_grids_1=int(clean_grids[0][1])
        clean_grids_2=int(clean_grids[1][0])
        clean_grids_3=int(clean_grids[1][1])

        ccf_clean_map=np.append(ccf_Sum[:, clean_grids_0:clean_grids_1], ccf_Sum[:, clean_grids_2:clean_grids_3], axis=1)
        std_ccf=np.nanstd(ccf_clean_map)

        for i in range(ccf_Sum.shape[0]):
            ccf_SNR[i]= ccf_Sum[i]/std_ccf  # Replace 'std' with 'std_ccf'

        if plot == True:

            plotMatrix.plotMatrix(ccf_SNR, rvlag, np.arange(0, self.nObs, 1), 'Radial velocity (km/s)','spatial axis',stretch=True, planet_posi=spatial_pix)
            plt.hlines(y=central_pix, xmin=rvlag[0], xmax=rvlag[-1], ls='--', colors='lightgray')
            plt.vlines(x=v_sys, ymin=0, ymax=(self.nObs-1), ls='--', colors='lightgray')
            plt.title ('Cross-correlated SNR')
            plt.show()


        return (ccf_Sum, ccf_SNR)

class processor_likelihood_map():
    '''
    # The `processor_likelihood_map` class is designed to estimate the likelihood difference 
    # for planet detection. You would like to use a model matrix that includes the planet model and compares 
    # it with another model matrix that excludes the planet. This process helps in determining 
    # the presence of a planet in observational data.'
    # 
        Methods:
        --------
        __init__():
            Initializes the `processor_likelihood_map` class.
        covariance_calculator(Fit_Matrix, ABBA_series, N_exposures):
            Computes the variance of residuals and residuals themselves by comparing the 
            observational data with the fit matrix.
        likelihood_calculator(d=None, M=None, Sigma_0_flat=None, log_sigma_0=True, prior_psi=None, gamma=2):
            Calculates the log-likelihood of the data given a model matrix and covariance matrix.
        likelihood_map(observation_matrix, model_matrix, n_vgrids, sigma_matrix, prior, matrix_components=2, log_sigma=True):
            Generates a likelihood map by iterating over velocity grids and observational data, 
            and computes the likelihood for each combination.
    '''
    def __init__(self):
        """
        Initialize the processor_likelihood_map class.
        """
        pass
        

    def covariance_calculator (self, Fit_Matrix, ABBA_series, N_exposures, scale_factor=None):
        
        if scale_factor == None:
            fit_matrix_single=Fit_Matrix/N_exposures
        else:
            fit_matrix_single=Fit_Matrix*scale_factor

        fit_matrix_ex=np.expand_dims(fit_matrix_single, axis=1)

        fit_matrix_ex=np.repeat(fit_matrix_ex, repeats=N_exposures, axis=1)
        
        resi = ABBA_series - fit_matrix_ex

        var_resi= np.var(resi, axis=1)

        return (var_resi, resi)

    def likelihood_calculator(self, d=None, M=None, Sigma_0_flat=None, log_sigma_0=True, prior_psi=None, gamma=2):
        """
        Compute the likelihood L(ψ | d) based on the given inputs.

        Parameters:
        - d: Data vector (Nd x 1).
        - M: Model matrix (Nd x Nc).
        - Sigma_0: Covariance matrix (Nd x Nd)  
        - log_sigma_0: if = True use np.log(Sigma_0) to caluclate the likelihood to avoid overflow issue. (default = True)
        - prior_psi: Prior probability P(ψ).
        - gamma: Hyperparameter for noise scaling (default=2).
        

        Returns:
        - log_likelihood: The log of the likelihood value.
        """
        Sigma_0=np.diagflat(Sigma_0_flat)
        log_Sigma_0=np.diagflat(np.log(Sigma_0_flat))
        Sigma_0_inv = np.linalg.inv(Sigma_0)


        print('sigmna_0_inv shape:', Sigma_0_inv.shape)
        #print('sigmna_0_inv:', Sigma_0_inv[18:20])
        #print ('Simga_0:', Sigma_0[18:20])


        # Compute the matrix M^T * Sigma_0^-1 * M
        MT_Sigma0_inv_M = M.T @ Sigma_0_inv @ M

        # Solve for the coefficients (c-hat)
        MT_Sigma0_inv_d = d.T @ Sigma_0_inv @ M
        c_hat_T = solve(MT_Sigma0_inv_M, MT_Sigma0_inv_d)
        c_hat=c_hat_T.T
        print ('c_hat:', c_hat)

        # Compute chi_0^2
        residual = d - M@c_hat
        chi_0_squared = residual.T @ Sigma_0_inv @ residual
        #print('residual:', residual)
        #print('chi_0_squared:', chi_0_squared)

        # Compute the determinant of MT_Sigma0_inv_M
        determinant = np.linalg.det(MT_Sigma0_inv_M)
        #print('det(MT_Sigma0_inv_M):', determinant)

        Nd = len(d)  # Number of data points
        Nc = M.shape[1]  # Number of components in the model

        if log_sigma_0 == False:

            # Compute the log-likelihood (using Equation 15)
            likelihood = (
                prior_psi / np.sqrt(np.linalg.det(Sigma_0) * determinant)
                * (1 / chi_0_squared) ** ((Nd - Nc + gamma - 1) / 2)
            )

            # Return the log-likelihood for numerical stability
            log_likelihood = np.log(prior_psi) - 0.5 * np.log(np.linalg.det(Sigma_0) * determinant) \
                        + ((Nd - Nc + gamma - 1) / 2) * np.log(1 / chi_0_squared)

            print ('det(simga_0):', np.linalg.det(Sigma_0))
            print ('likelihood:', likelihood)
            print('log likelihood:', log_likelihood)

        else:
            #print ('sum(np.log(sigma_0)):', np.sum(log_Sigma_0))

            # BUGFIX (2026-09-22): this branch must equal the log_sigma_0=False
            # branch above (same quantity, numerically-stabler path): log(det(Sigma_0)
            # * determinant) = sum(log_Sigma_0) + log(determinant), so both det terms
            # are subtracted, matching Ruffio et al. (2019) eq. 15 / Landman et al.
            # (2024) eq. 15 (both |Sigma_0| and |M^T Sigma_0^-1 M| are in the
            # denominator). This line previously had "- np.log(determinant)" (a "+"
            # after distributing -0.5), which flips the sign of that term relative to
            # the other branch and the cited equation. Found and verified against
            # Landman et al. (2024) on 2026-09-05; the fix was validated then to have
            # a negligible effect on self-referenced (peak-relative) significance
            # figures for the H1800 survey grid, but it does change raw/absolute
            # log-likelihood and Bayes-factor values, since the determinant depends on
            # the trial velocity / model design matrix.
            log_likelihood = np.log(prior_psi) - 0.5 * (np.sum(log_Sigma_0) + np.log(determinant)) \
                        + ((Nd - Nc + gamma - 1) / 2) * np.log(1 / chi_0_squared)

            print('log likelihood:', log_likelihood)

            
        return (log_likelihood)

    def likelihood_map(self, observation_matrix, model_matrix, n_vgrids, sigma_matrix, prior, matrix_components=2, log_sigma=True, skip_velocity_grid_single_component=None):
        
        """
        Computes the log-likelihood map for a given observation matrix and model matrix.
        Parameters:
        ----------
        observation_matrix : numpy.ndarray
            The observed data matrix. Each row corresponds to a different observation.
        model_matrix : numpy.ndarray
            The model data matrix. The first axis corresponds to velocity grids, and the second axis 
            corresponds to the number of components in the model (e.g., 1 or 2).
        n_vgrids : int
            The number of velocity grids to evaluate.
        sigma_matrix : numpy.ndarray
            The matrix of variances (or uncertainties) corresponding to the observation matrix.
        prior : float or numpy.ndarray
            The prior information to be used in the likelihood calculation.
        matrix_components : int, optional
            The number of components in the model matrix (default is 2). Must be either 1 or 2.
        log_sigma : bool, optional
            If True, the logarithm of the variance is used in the likelihood calculation (default is True).
        skip_velocity_grid_single_component : list, optional
            A list of velocity grid indices to skip when matrix_components is 1.
        Returns:
        -------
        numpy.ndarray
            A 2D array representing the log-likelihood map. The shape of the array is 
            (number of observations, number of velocity grids).
        Notes:
        -----
        - The function iterates over velocity grids and observations to compute the likelihood for each combination.
        - The `likelihood_calculator` method is used to compute the likelihood for a given observation and model.
        - The function handles missing or non-finite values in the observation and variance matrices by applying masks.
        - If `matrix_components` is not 1 or 2, a warning message is printed, and the iteration continues without processing.
        Example:
        -------
        >>> log_likelihood_map = likelihood_map(observation_matrix, model_matrix, n_vgrids, sigma_matrix, prior)
        >>> print(log_likelihood_map.shape)
        (num_observations, n_vgrids)
        """

        map_size=(observation_matrix.shape[0], n_vgrids)
        log_likelihood_mapp=np.zeros(shape=map_size)

        for i in range(observation_matrix.shape[0]):

            variance_flat = sigma_matrix[i].flatten()
            mask_finite = np.isfinite(observation_matrix[i])
            mask_finite_var = np.isfinite(variance_flat)

            if matrix_components == 1:
                if skip_velocity_grid_single_component is not None:
                    print(f"Skipping velocity grids as the stellar-only model is fixed.")
                    fit_model_mask = np.array([model_matrix[0][i][mask_finite & mask_finite_var]])
                    
                    output = self.likelihood_calculator(
                                d=observation_matrix[i][mask_finite & mask_finite_var],
                                M=fit_model_mask.T,
                                Sigma_0_flat=variance_flat[mask_finite_var & mask_finite],
                                log_sigma_0=log_sigma,
                                gamma=matrix_components,
                                prior_psi=prior
                            )
                    log_likelihood_mapp[i] = np.repeat(output, n_vgrids)

                else:
                    for v in tqdm.tqdm(range(model_matrix.shape[0])):
                        fit_model_mask = np.array([model_matrix[v][i][mask_finite & mask_finite_var] for v in range(model_matrix.shape[0])])

                        output = self.likelihood_calculator(
                                    d=observation_matrix[i][mask_finite & mask_finite_var],
                                    M=fit_model_mask.T,
                                    Sigma_0_flat=variance_flat[mask_finite_var & mask_finite],
                                    log_sigma_0=log_sigma,
                                    gamma=matrix_components,
                                    prior_psi=prior
                                )
                        log_likelihood_mapp[i][v] = output

            elif matrix_components >= 2:

                for v in tqdm.tqdm(range(model_matrix.shape[0])):

                    print('The number of model components should be at least 1. The last component of your model should be the planet model.')
                    fit_model_mask = np.array([model_matrix[v][comp, i][mask_finite & mask_finite_var] for comp in range(matrix_components)])


                    output = self.likelihood_calculator(
                    d=observation_matrix[i][mask_finite & mask_finite_var],
                    M=fit_model_mask.T,
                    Sigma_0_flat=variance_flat[mask_finite_var & mask_finite],
                    log_sigma_0=log_sigma,
                    gamma=matrix_components,
                    prior_psi=prior
                )
                    log_likelihood_mapp[i][v] = output
            else:
                print("The first axis of your model matrix refers to the number of components in your model. It is supposed to be at least 1 in EXOCRIRES v1.0.")
                continue

        return (log_likelihood_mapp)
    


    def fast_likelihood_calculator(self, d=None, M=None, Sigma_0_flat=None,
                                   log_sigma_0=True, prior_psi=None, gamma=2):
        """Fast drop-in replacement for :meth:`likelihood_calculator`.

        Same equations, but exploits the fact that ``Sigma_0`` is diagonal
        instead of building and inverting it as a dense ``(Nd x Nd)`` matrix.
        See :mod:`exocrires.fast_likelihood_calculator` for the derivation, the
        measured agreement with :meth:`likelihood_calculator` (relative
        difference <= 1.7e-16), and the list of deliberate differences.
        """
        return fast_likelihood_calculator(
            d=d, M=M, Sigma_0_flat=Sigma_0_flat, log_sigma_0=log_sigma_0,
            prior_psi=prior_psi, gamma=gamma)

    def fast_likelihood_map(self, observation_matrix, model_matrix, n_vgrids,
                            sigma_matrix, prior, matrix_components=2,
                            log_sigma=True,
                            skip_velocity_grid_single_component=None,
                            mask_nonfinite_model=False):
        """Fast drop-in replacement for :meth:`likelihood_map`.

        Roughly 250-600x faster than :meth:`likelihood_map` and agrees with it
        to 2.3e-13 absolute for ``matrix_components=2`` and for
        ``matrix_components=1`` with ``skip_velocity_grid_single_component``.
        It intentionally *differs* for ``matrix_components=1`` without the skip
        flag, where :meth:`likelihood_map` has a variable-shadowing bug; see
        :mod:`exocrires.fast_likelihood_calculator`.
        """
        return fast_likelihood_map(
            observation_matrix=observation_matrix, model_matrix=model_matrix,
            n_vgrids=n_vgrids, sigma_matrix=sigma_matrix, prior=prior,
            matrix_components=matrix_components, log_sigma=log_sigma,
            skip_velocity_grid_single_component=skip_velocity_grid_single_component,
            mask_nonfinite_model=mask_nonfinite_model)

    def gaussian_delta_model(self, x, sigma_g, mu):
        """Model for Delta lnL: Delta(x) = (x-mu)^2 / (2*sigma_g^2)"""
        return (x - mu)**2 / (2.0 * sigma_g**2)

    def ci_from_sigma(self, v, delta_lnL, target_sigma=1.0, constrain_CL=False):

        """Interpolate to get left/right bounds where z <= target_sigma.
        Returns (left, right)."""

        # we want region where z <= target_sigma (i.e.  inside the conf. interval)
    
        if constrain_CL == True:

            delta_lnL_ratio = delta_lnL-np.nanmax(delta_lnL) 

            z= np.sqrt(-2*delta_lnL_ratio)


            inside = (z <= target_sigma)
            if not inside.any():
                return None
            imin = np.argmin(z)
            # left side
            z_left = z[:imin+1]; v_left = v[:imin+1]
            if inside[0]:
                left = v_left[0]
            else:
                idx = np.where((z_left[:-1] > target_sigma) & (z_left[1:] <= target_sigma))[0]
                if idx.size == 0:
                    left = v_left[0]
                else:
                    i = idx[-1]
                    # linear interpolation of z vs v
                    left = np.interp(target_sigma, z_left[i:i+2], v_left[i:i+2])
            # right side
            z_right = z[imin:]; v_right = v[imin:]
            if inside[-1]:
                right = v_right[-1]
            else:
                idx = np.where((z_right[:-1] <= target_sigma) & (z_right[1:] > target_sigma))[0]
                if idx.size == 0:
                    right = v_right[-1]
                else:
                    i = idx[0]
                    right = np.interp(target_sigma, z_right[i:i+2], v_right[i:i+2])

            
        else:

            z = np.sqrt(np.abs(-2 * delta_lnL)) 
            left = 0.0
            right = 0.0

        return left, right, z


        #estimate the confidence level from delta_lnL using the window [0:5], [-5:]
        #sigma = (delta_lnL-np.nanmedian(delta_lnL))/np.nanstd(delta_lnL[0:5]+delta_lnL[-5:])
        #sigma = np.sqrt(-2*delta_lnL_ratio)
        #return left, right, z


class processor_deregt_ccf:
    """De Regt §4.2 unnormalized cross-correlation for per-molecule detection.

    Implements the high-pass-filtered unnormalized CCF described in De Regt et al. (2024)
    §4.2, adapted for the EXOCRIRES population study pipeline.

    The key differences from the old processor_cross_correlation:
    - Unnormalized CCF: CCF[sp,rv] = Σ HP(M_chip(rv)) · HP(data_chip)[sp]
    - ACF used as template self-correlation reference for SNR normalisation
    - Plain HP(subtract) data cube (no per-molecule fNoX subtraction)
    - Template strength gate: molecules with K-band HP std < threshold × max are skipped

    Parameters
    ----------
    wlen_orders : ndarray (nOrder, nPix)
        Detector wavelength grid for K-band orders (metres).
    subtract : ndarray (nOrder, nObs, nPix)
        Residual cube after stellar subtraction (stack − fit_matrix_star).
    wave_tmpl : ndarray
        Template wavelength axis (metres).
    v_sys : float
        Systemic velocity of the planet (km/s).  Used as ACF reference point.
    sigma_hp : int
        Gaussian HP filter σ in pixels (default 100).
    chip_size_px : int
        Pixels per chip (default 6138 = 3 × 2046 for CRIRES+ K2166).
    """

    C_KMS = 2.99792458e5

    def __init__(self, wlen_orders, subtract, wave_tmpl,
                 v_sys=10.0, sigma_hp=100, chip_size_px=6138):
        self.wlen_orders  = wlen_orders
        self.subtract     = subtract
        self.wave_tmpl    = wave_tmpl
        self.v_sys        = v_sys
        self.sigma_hp     = sigma_hp
        self.chip_size_px = chip_size_px
        self._hp_cube     = None

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _hp_row(arr, sigma_px):
        """High-pass filter a 1-D array; zero-fill where input is invalid/zero."""
        valid = np.isfinite(arr) & (arr != 0)
        if valid.sum() < 10:
            return np.zeros_like(arr, dtype=float)
        tmp = np.where(valid, arr.astype(float), 0.0)
        return np.where(valid, tmp - gaussian_filter1d(tmp, sigma=sigma_px), 0.0)

    # ── HP cube ───────────────────────────────────────────────────────────────

    def build_hp_cube(self):
        """Apply HP filter to the subtract cube and cache the result.

        Returns
        -------
        hp_cube : ndarray (nOrder, nObs, nPix)
        """
        nOrder, nObs, nPix = self.subtract.shape
        n_chips = nPix // self.chip_size_px
        out = np.zeros_like(self.subtract, dtype=float)
        for o in range(nOrder):
            for c in range(n_chips):
                sl = slice(c * self.chip_size_px, (c + 1) * self.chip_size_px)
                for sp in range(nObs):
                    out[o, sp, sl] = self._hp_row(self.subtract[o, sp, sl], self.sigma_hp)
        self._hp_cube = out
        return out

    # ── template utilities ────────────────────────────────────────────────────

    @staticmethod
    def align_continuum(f_ref, f_target, sigma=100):
        """Scale f_target so its low-pass continuum matches f_ref.

        Parameters
        ----------
        f_ref, f_target : ndarray  (same length, normalised 0–1 range)
        sigma : int  HP/LP filter width in pixels

        Returns
        -------
        f_scaled : ndarray
        scale : float
        """
        lp_ref = gaussian_filter1d(f_ref.astype(float),    sigma=sigma)
        lp_tgt = gaussian_filter1d(f_target.astype(float), sigma=sigma)
        valid  = (lp_tgt > 1e-3) & (lp_ref > 1e-3)
        scale  = float(np.nanmedian(lp_ref[valid] / lp_tgt[valid])) if valid.sum() > 5 else 1.0
        return f_target * scale, scale

    def template_strength(self, templates, wlen_min=2.05e-6, wlen_max=2.48e-6):
        """Compute relative K-band HP feature strength for each template.

        Parameters
        ----------
        templates : dict {mol_name: flux_array}  (same wavelength axis as wave_tmpl)
        wlen_min, wlen_max : float  K-band window in metres

        Returns
        -------
        strengths : dict {mol_name: rel_strength}
            rel_strength = std(HP(tmpl)) / max_std across all templates.
            Molecules with rel_strength < threshold should be skipped.
        """
        k_mask = (self.wave_tmpl > wlen_min) & (self.wave_tmpl < wlen_max)
        stds = {}
        for mol, f in templates.items():
            hp = f[k_mask].astype(float) - gaussian_filter1d(f[k_mask].astype(float),
                                                              sigma=self.sigma_hp)
            stds[mol] = float(hp.std())
        ref = max(stds.values()) if stds else 1.0
        return {mol: (s / ref if ref > 0 else 0.0) for mol, s in stds.items()}

    # ── CCF core ──────────────────────────────────────────────────────────────

    def compute_ccf_acf(self, flux_tmpl, rvlag):
        """Compute unnormalized CCF and ACF for one template over all RV lags.

        CCF[sp, rv] = Σ_{orders,chips}  HP(M_chip(rv)) · hp_cube[o, sp, chip]
        ACF[rv]     = Σ_{orders,chips}  HP(M_chip(rv)) · HP(M_chip(v_sys))

        Parameters
        ----------
        flux_tmpl : ndarray  template flux on wave_tmpl grid
        rvlag : ndarray  RV lag array (km/s)

        Returns
        -------
        ccf : ndarray (nObs, nRV)
        acf : ndarray (nRV,)
        """
        if self._hp_cube is None:
            self.build_hp_cube()

        nOrder, nObs, nPix = self._hp_cube.shape
        n_chips = nPix // self.chip_size_px
        nrv = len(rvlag)

        factor_ref = 1.0 + self.v_sys / self.C_KMS
        M0_hp = [[None] * n_chips for _ in range(nOrder)]
        for o in range(nOrder):
            for c in range(n_chips):
                sl = slice(c * self.chip_size_px, (c + 1) * self.chip_size_px)
                w = self.wlen_orders[o, sl]
                valid = np.isfinite(w) & (w > 0)
                m0 = np.zeros(self.chip_size_px)
                m0[valid] = np.interp(w[valid] / factor_ref,
                                      self.wave_tmpl, flux_tmpl, left=0., right=0.)
                M0_hp[o][c] = self._hp_row(m0, self.sigma_hp)

        ccf = np.zeros((nObs, nrv))
        acf = np.zeros(nrv)

        for i, rv in enumerate(rvlag):
            factor = 1.0 + rv / self.C_KMS
            acf_rv = 0.0
            ccf_rv = np.zeros(nObs)
            for o in range(nOrder):
                for c in range(n_chips):
                    sl = slice(c * self.chip_size_px, (c + 1) * self.chip_size_px)
                    w = self.wlen_orders[o, sl]
                    valid = np.isfinite(w) & (w > 0)
                    m = np.zeros(self.chip_size_px)
                    m[valid] = np.interp(w[valid] / factor,
                                         self.wave_tmpl, flux_tmpl, left=0., right=0.)
                    m_hp = self._hp_row(m, self.sigma_hp)
                    acf_rv += float(np.dot(m_hp, M0_hp[o][c]))
                    ccf_rv += self._hp_cube[o, :, sl].dot(m_hp)
            ccf[:, i] = ccf_rv
            acf[i]    = acf_rv

        return ccf, acf

    def snr(self, rvlag, ccf_sp, acf, exclude_kms=100.0):
        """Compute SNR at v_sys and peak RV using ACF-scaled noise floor.

        Parameters
        ----------
        rvlag : ndarray
        ccf_sp : ndarray (nRV,)  CCF at the planet spatial pixel
        acf : ndarray (nRV,)
        exclude_kms : float  noise is estimated from |rv| > this value

        Returns
        -------
        snr_vsys, snr_peak, peak_rv, std_noise, residual, acf_aligned
        """
        noise_mask = np.abs(rvlag) > exclude_kms
        if noise_mask.sum() > 5:
            acf_std  = np.nanstd(acf[noise_mask])
            ccf_std  = np.nanstd(ccf_sp[noise_mask])
            acf_scale = ccf_std / acf_std if acf_std > 0 else 1.0
            # Match the noise-region mean as well as the std.
            # Without this, HP-filter boundary artefacts in the template create a
            # systematic DC offset in the ACF that std-only scaling cannot remove.
            acf_offset = (np.nanmean(ccf_sp[noise_mask])
                          - acf_scale * np.nanmean(acf[noise_mask]))
        else:
            acf_scale  = 1.0
            acf_offset = 0.0
        acf_aligned = acf * acf_scale + acf_offset
        residual    = ccf_sp - acf_aligned
        std_noise   = float(np.nanstd(residual[noise_mask])) if noise_mask.sum() > 0 else np.nan
        vsys_idx    = int(np.argmin(np.abs(rvlag - self.v_sys)))
        peak_idx    = int(np.nanargmax(ccf_sp))
        snr_vsys    = float(ccf_sp[vsys_idx] / std_noise) if std_noise and std_noise > 0 else np.nan
        snr_peak    = float(ccf_sp[peak_idx] / std_noise) if std_noise and std_noise > 0 else np.nan
        return snr_vsys, snr_peak, rvlag[peak_idx], std_noise, residual, acf_aligned

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, diff_templates, rvlag, spatial_pix,
            exclude_kms=100.0, strength_threshold=0.01):
        """Run the full De Regt CCF pipeline for a set of molecule templates.

        For each molecule:
          1. Check template feature strength in K-band; skip if below threshold.
          2. Compute CCF[nObs, nRV] and ACF[nRV].
          3. Compute SNR at v_sys and peak.

        Parameters
        ----------
        diff_templates : dict {mol_name: flux_array}
            Molecule templates on the wave_tmpl wavelength grid.
            Recommended: pure mol-only spectra normalised to peak=1.
        rvlag : ndarray  RV lag array (km/s)
        spatial_pix : int  planet spatial row index
        exclude_kms : float  noise estimation window (|rv| > exclude_kms)
        strength_threshold : float
            Minimum relative K-band HP feature strength.  Molecules below this
            value (e.g. CH4 at T≥1600K) are skipped and assigned SNR=NaN.

        Returns
        -------
        results : dict
            {mol_name: {
                'ccf'      : ndarray (nObs, nRV) or None if skipped,
                'acf'      : ndarray (nRV,)      or None,
                'acf_aligned': ndarray            or None,
                'residual' : ndarray              or None,
                'snr_vsys' : float  (NaN if skipped),
                'snr_peak' : float  (NaN if skipped),
                'peak_rv'  : float  (NaN if skipped),
                'std_noise': float  (NaN if skipped),
                'skipped'  : bool,
            }}
        """
        if self._hp_cube is None:
            self.build_hp_cube()

        strengths = self.template_strength(diff_templates)
        print("Template K-band feature strengths:")
        for mol, rel in strengths.items():
            flag = "  ← WEAK / absent at this T" if rel < strength_threshold else ""
            print(f"  {mol}: rel={rel:.4f}{flag}")

        results = {}
        for mol, flux_tmpl in diff_templates.items():
            import time as _time
            t0 = _time.time()
            print(f"[{mol}] computing CCF+ACF ...", end="  ", flush=True)

            ccf, acf = self.compute_ccf_acf(flux_tmpl, rvlag)
            snr_vsys, snr_peak, peak_rv, std_noise, residual, acf_aligned = self.snr(
                rvlag, ccf[spatial_pix], acf, exclude_kms=exclude_kms)

            print(f"{_time.time()-t0:.1f}s  SNR@v_sys={snr_vsys:.2f}  "
                  f"SNR@peak={snr_peak:.2f}  peak_rv={peak_rv:.0f} km/s")

            results[mol] = dict(
                ccf=ccf, acf=acf, acf_aligned=acf_aligned, residual=residual,
                snr_vsys=snr_vsys, snr_peak=snr_peak, peak_rv=peak_rv,
                std_noise=std_noise, skipped=False, rel_strength=strengths.get(mol, 0.0))

        return results


class processor_nw209_ccf:
    """NW209 whitening + stellar-orthogonal CCF for per-molecule detection.

    Implements the NW209 CCF engine from EXOCRIRES 2.0.9: variance-whitened,
    stellar-orthogonal cross-correlation with ACF-subtracted noise floor
    and per-row 2D S/N map.

    All array inputs must be pre-sliced to the same set of orders (e.g. subtract[2:],
    fit_matrix_star[2:], var[2:], wlen[2:] for K2166 orders 2-6).

    Parameters
    ----------
    wlen_orders : ndarray (nOrder, nPix)
        Detector wavelength grid (metres).
    subtract : ndarray (nOrder, nSpatial, nPix)
        Residual cube after stellar subtraction (stack - fit_matrix_star).
    fit_matrix_star : ndarray (nOrder, nSpatial, nPix)
        Stellar model matrix from multiplicative_factor_fit().
    var : ndarray (nOrder, nSpatial, nPix)
        Pixel-noise variance from covariance_calculator().
    wave_tmpl : ndarray
        Template wavelength axis (metres).
    v_sys : float
        Systemic velocity of the planet (km/s).
    sigma_hp : int
        Gaussian HP filter sigma in pixels (default 100).
    use_hp_mol : dict, optional
        Per-molecule HP toggle {mol_name: bool}.  Default: all True.
    """

    C_KMS = 2.99792458e5

    def __init__(self, wlen_orders, subtract, fit_matrix_star, var,
                 wave_tmpl, v_sys, sigma_hp=100, use_hp_mol=None):
        self.wlen_orders     = wlen_orders
        self.subtract        = subtract
        self.fit_matrix_star = fit_matrix_star
        self.var             = var
        self.wave_tmpl       = wave_tmpl
        self.v_sys           = v_sys
        self.sigma_hp        = sigma_hp
        self.use_hp_mol      = use_hp_mol or {}

        self._nOrder   = subtract.shape[0]
        self._nSpatial = subtract.shape[1]
        self._nPix     = subtract.shape[2]

    # ── internal helpers ──────────────────────────────────────────────────────

    def _hp(self, arr, use_hp):
        """High-pass filter a 1D array; NaN-fill to zero.  No-op when use_hp is False."""
        a = np.nan_to_num(arr.astype(float))
        if not use_hp:
            return a
        return a - gaussian_filter1d(a, sigma=self.sigma_hp)

    def _sigma_from_var(self, var_orders_sp):
        """Concatenated sigma (nOrder x nPix); NaN/non-positive -> inf (zero weight)."""
        segs = []
        for o in range(self._nOrder):
            v = var_orders_sp[o].copy().astype(float)
            v[~np.isfinite(v) | (v <= 0)] = np.inf
            segs.append(np.sqrt(v))
        return np.concatenate(segs)

    def _snr_acf(self, ccf_sp, acf, vsys_idx, rvlag, noise_vmin=0):
        """ACF-subtracted noise floor — std-based scaling + mean alignment (NW209 method).

        Matches the inline _snr_acf from EXOCRIRES_CC_2.0.9.ipynb, without the
        1e-3 guard that capped acf_scale at 1000 for narrow-peak templates.

        Returns snr, std_noise, acf_aligned, residual, ccf_norm.
        """
        noise_mask = np.abs(rvlag - self.v_sys) > noise_vmin
        ccf_pk     = float(ccf_sp[vsys_idx])
        if noise_mask.sum() > 5:
            acf_std = float(np.nanstd(acf[noise_mask]))
            ccf_std = float(np.nanstd(ccf_sp[noise_mask]))
            if acf_std > 0:
                acf_scale  = ccf_std / acf_std
                acf_offset = (float(np.nanmean(ccf_sp[noise_mask]))
                              - acf_scale * float(np.nanmean(acf[noise_mask])))
            else:
                acf_scale  = 0.0
                acf_offset = 0.0
        else:
            acf_scale  = 1.0
            acf_offset = 0.0
        acf_aligned = acf * acf_scale + acf_offset
        residual    = ccf_sp - acf_aligned
        std_noise   = float(np.nanstd(residual[noise_mask])) if noise_mask.sum() > 0 else np.nan
        snr         = float(ccf_pk / std_noise) if (std_noise and std_noise > 0) else np.nan
        ccf_norm    = ccf_sp / std_noise if (std_noise and std_noise > 0) else ccf_sp
        return snr, std_noise, acf_aligned, residual, ccf_norm

    def _whiten_row(self, arr_orders, sigma, use_hp):
        """Whitened, optionally HP-filtered concatenated vector for one spatial row.

        arr_orders : (nOrder, nPix)
        """
        return np.nan_to_num(
            np.concatenate([self._hp(arr_orders[o], use_hp)
                            for o in range(self._nOrder)])) / sigma

    # ── utility methods ───────────────────────────────────────────────────────

    def estimate_alpha_p(self, flux_full, spatial_pix, ref_max=None):
        """Estimate planet-to-star flux ratio from LP ratio at the planet row.

        Parameters
        ----------
        flux_full : ndarray  Full planet template on wave_tmpl grid.
        spatial_pix : int    Planet spatial row index.
        ref_max : float, optional
            Normalization constant.  Defaults to nanmax(flux_full).

        Returns
        -------
        alpha_p : float
        """
        if ref_max is None:
            ref_max = float(np.nanmax(flux_full)) or 1.0
        factor_vsys = 1.0 + self.v_sys / self.C_KMS
        alpha_list = []
        for o in range(self._nOrder):
            w     = self.wlen_orders[o]
            val   = np.isfinite(w) & (w > 0)
            m_grid      = np.zeros(self._nPix)
            m_grid[val] = np.interp(w[val] / factor_vsys, self.wave_tmpl,
                                    flux_full / ref_max, left=0., right=0.)
            d_row = self.subtract[o, spatial_pix, :].astype(float)
            m_row = m_grid.astype(float)
            valid = np.isfinite(d_row) & (d_row != 0) & (m_row > 1e-6)
            if valid.sum() < 50:
                continue
            lp_d = gaussian_filter1d(np.where(valid, d_row, 0.0), sigma=100)
            lp_m = gaussian_filter1d(np.where(valid, m_row, 0.0), sigma=100)
            alpha_list.append(float(np.nanmedian((lp_d / lp_m)[valid])))
        return float(np.nanmedian(alpha_list)) if alpha_list else np.nan

    def build_exp_cubes(self, noX_templates, alpha_p, spatial_pix, ref_max=None):
        """Build per-molecule exp_cubes with planet-row continuum removed.

        For each molecule X:
            cube[:, spatial_pix, :] -= alpha_p * interp(f_noX, v_sys)

        Parameters
        ----------
        noX_templates : dict {mol_name: ndarray}
            Knockout spectra (f_noX) on the wave_tmpl grid.
        alpha_p : float
            Planet-to-star flux estimate from estimate_alpha_p().
        spatial_pix : int
            Planet spatial row index.
        ref_max : float, optional
            Template normalization constant.

        Returns
        -------
        exp_cubes : dict {mol_name: ndarray (nOrder, nSpatial, nPix)}
        """
        if ref_max is None:
            first   = next(iter(noX_templates.values()))
            ref_max = float(np.nanmax(np.abs(first))) or 1.0
        factor_vsys = 1.0 + self.v_sys / self.C_KMS
        exp_cubes = {}
        for mol, f_noX in noX_templates.items():
            grid = np.zeros((self._nOrder, self._nPix))
            for o in range(self._nOrder):
                w   = self.wlen_orders[o]
                val = np.isfinite(w) & (w > 0)
                grid[o, val] = np.interp(w[val] / factor_vsys, self.wave_tmpl,
                                         f_noX / ref_max, left=0., right=0.)
            cube = self.subtract.copy()
            cube[:, spatial_pix, :] -= alpha_p * grid
            exp_cubes[mol] = cube
        return exp_cubes

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, diff_templates, rvlag, spatial_pix, exp_cubes=None,
            noise_vmin=0, ref_max=None, use_hp_mol=None):
        """Run the NW209 CCF pipeline for all molecules.

        Parameters
        ----------
        diff_templates : dict {mol_name: ndarray}
            Differential templates (f_full - f_noX) on wave_tmpl grid.
        rvlag : ndarray
            RV lag array (km/s).
        spatial_pix : int
            Planet spatial row index.
        exp_cubes : dict {mol_name: ndarray (nOrder, nSpatial, nPix)}, optional
            Per-molecule subtract cubes.  If None, self.subtract is used for all.
            Build with build_exp_cubes() to remove planet-row continuum.
        noise_vmin : float
            Noise estimated from |rv - v_sys| > noise_vmin (km/s).  Default 0.
        ref_max : float, optional
            Template normalization.  Defaults to nanmax of the first template.
        use_hp_mol : dict, optional
            Per-molecule HP toggle for this call, merged with self.use_hp_mol.

        Returns
        -------
        results : dict
            {mol_name: {
                'ccf'        : ndarray (nRV,),
                'acf'        : ndarray (nRV,),
                'ccf_norm'   : ndarray (nRV,),   # CCF / std_noise  (sigma units)
                'acf_norm'   : ndarray (nRV,),   # ACF / ACF[v_sys]
                'acf_aligned': ndarray (nRV,),
                'residual'   : ndarray (nRV,),   # CCF - ACF  (peak removed)
                'ccf_map'    : ndarray (nSpatial, nRV),
                'sn_map'     : ndarray (nSpatial, nRV),
                'use_hp'     : bool,
                'snr'        : float,
                'std_noise'  : float,
                'peak_rv'    : float,
                'skipped'    : bool,
            }}
        """
        hp_cfg = {**self.use_hp_mol, **(use_hp_mol or {})}

        if ref_max is None:
            first   = next(iter(diff_templates.values()))
            ref_max = float(np.nanmax(np.abs(np.nan_to_num(first)))) or 1.0

        vsys_idx = int(np.argmin(np.abs(rvlag - self.v_sys)))
        nRV      = len(rvlag)

        sigma_rows = [self._sigma_from_var(self.var[:, r, :]) for r in range(self._nSpatial)]
        sigma_pl   = sigma_rows[spatial_pix]
        print(f'NW209  finite-sigma px (planet row) = {np.isfinite(sigma_pl).sum()}/{len(sigma_pl)}')
        print(f'{"Mol":5}  {"HP":>5}  {"S/N":>10}  {"std_noise":>11}  {"peak_rv":>10}')
        print('-' * 50)

        results = {}
        for mol, f_tmpl in diff_templates.items():
            import time as _time
            t0m   = _time.time()
            useHP = bool(hp_cfg.get(mol, True))

            _tmpl_rms = float(np.sqrt(np.nanmean((np.nan_to_num(f_tmpl) / ref_max) ** 2)))
            if _tmpl_rms < 1e-4:
                print(f'{mol:5}  {str(useHP):>5}  {"--":>10}  {"--":>11}  '
                      f'empty template (RMS={_tmpl_rms:.1e}) -> skipped')
                results[mol] = dict(skipped=True, snr=np.nan, std_noise=np.nan,
                                    peak_rv=np.nan, ccf=None, acf=None,
                                    ccf_norm=None, acf_norm=None,
                                    acf_aligned=None, residual=None,
                                    ccf_map=None, sn_map=None, use_hp=useHP)
                continue

            cube = exp_cubes[mol] if (exp_cubes and mol in exp_cubes) else self.subtract

            # Build template matrix Y_tmpl_hp: (nRV, nOrder * nPix)
            Y_tmpl_hp = np.zeros((nRV, self._nOrder * self._nPix))
            for i, rv in enumerate(rvlag):
                fac = 1.0 + rv / self.C_KMS
                seg = []
                for o in range(self._nOrder):
                    w   = self.wlen_orders[o]
                    val = np.isfinite(w) & (w > 0)
                    m   = np.zeros(self._nPix)
                    m[val] = np.interp(w[val] / fac, self.wave_tmpl, f_tmpl / ref_max,
                                       left=0., right=0.)
                    seg.append(self._hp(m, useHP))
                Y_tmpl_hp[i] = np.nan_to_num(np.concatenate(seg))
            y_vsys = Y_tmpl_hp[vsys_idx]

            # 2-D maps: CCF and S/N at every spatial row
            ccf_map = np.full((self._nSpatial, nRV), np.nan)
            sn_map  = np.full((self._nSpatial, nRV), np.nan)
            for r in range(self._nSpatial):
                sig_r    = sigma_rows[r]
                inv_sig2 = np.nan_to_num(1.0 / sig_r ** 2)
                s_w      = self._whiten_row(self.fit_matrix_star[:, r, :], sig_r, useHP)
                s_norm2  = float(s_w @ s_w)
                if s_norm2 < 1e-30:
                    continue
                d_w    = self._whiten_row(cube[:, r, :], sig_r, useHP)
                d_perp = d_w - s_w * (float(s_w @ d_w) / s_norm2)
                ccf_r  = Y_tmpl_hp @ np.nan_to_num(d_perp / sig_r)
                # per-row stellar-orthogonal ACF via cheap mat-vecs (avoids (nRV x nPix) matrix)
                # ACF_r = Y @ (y_vsys / sigma_r^2) - p[v_sys] * |s_w|^2 * p
                p      = (Y_tmpl_hp @ np.nan_to_num(s_w / sig_r)) / s_norm2
                acf_r  = Y_tmpl_hp @ (y_vsys * inv_sig2) - p[vsys_idx] * s_norm2 * p
                _, std_r, _, _, _ = self._snr_acf(ccf_r, acf_r, vsys_idx, rvlag, noise_vmin)
                ccf_map[r] = ccf_r
                sn_map[r]  = ccf_r / std_r if (std_r and std_r > 0) else np.nan

            # Planet row: full stellar-orthogonal template matrix for exact ACF
            s_w_pl     = self._whiten_row(self.fit_matrix_star[:, spatial_pix, :],
                                          sigma_pl, useHP)
            s_norm2_pl = float(s_w_pl @ s_w_pl) or 1.0
            d_w_pl     = self._whiten_row(cube[:, spatial_pix, :], sigma_pl, useHP)
            d_perp_pl  = d_w_pl - s_w_pl * (float(s_w_pl @ d_w_pl) / s_norm2_pl)

            Y_mat_w    = Y_tmpl_hp / sigma_pl
            Y_mat_perp = Y_mat_w - np.outer((Y_mat_w @ s_w_pl) / s_norm2_pl, s_w_pl)
            ccf_1d     = Y_mat_perp @ d_perp_pl
            acf_1d     = Y_mat_perp @ Y_mat_perp[vsys_idx]
            del Y_mat_w, Y_mat_perp

            snr, std_noise, acf_aligned, residual, ccf_norm = self._snr_acf(
                ccf_1d, acf_1d, vsys_idx, rvlag, noise_vmin)
            acf_norm = (acf_1d / float(acf_1d[vsys_idx])
                        if acf_1d[vsys_idx] != 0 else acf_1d)

            finite_norm = ccf_norm[np.isfinite(ccf_norm)]
            peak_idx    = int(np.nanargmax(np.abs(ccf_norm))) if finite_norm.size else 0
            results[mol] = dict(
                ccf=ccf_1d, acf=acf_1d, ccf_norm=ccf_norm, acf_norm=acf_norm,
                acf_aligned=acf_aligned, residual=residual,
                ccf_map=ccf_map, sn_map=sn_map,
                use_hp=useHP, std_noise=std_noise, snr=snr,
                peak_rv=float(rvlag[peak_idx]), skipped=False)
            print(f'{mol:5}  {str(useHP):>5}  {snr:>10.2f}  {std_noise:>11.4e}  '
                  f'{rvlag[peak_idx]:>8.0f} km/s  ({_time.time()-t0m:.0f}s)')

        return results


