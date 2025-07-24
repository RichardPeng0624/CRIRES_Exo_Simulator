import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

from scipy import interpolate
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.linalg import svd, solve, det
from scipy.signal import savgol_filter

from exocrires import info
from exocrires import plotMatrix 

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

    def multiplicative_factor_fit (self, planet_model_spectrum, order_list, window_len, Polyorder):

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
                # Integrate the master spectrum over the spatial axis
                master_int = np.trapz(self.spectra[i, 16:19, x*2046:(x+1)*2046], axis=0)
                mask = np.isfinite(master_int)

                # Loop through each observation
                for j in range(self.spectra.shape[1]): 
                    mask_data = np.isfinite(self.spectra[i, j][x*2046:(x+1)*2046])

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
        

    def covariance_calculator (self, Fit_Matrix, ABBA_series, N_exposures):
        
        
        fit_matrix_single=Fit_Matrix/N_exposures

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
        
            log_likelihood = np.log(prior_psi) - 0.5 * (np.sum(log_Sigma_0) - np.log(determinant)) \
                        + ((Nd - Nc + gamma - 1) / 2) * np.log(1 / chi_0_squared)

            print('log likelihood:', log_likelihood)

            
        return (log_likelihood)

    def likelihood_map(self, observation_matrix, model_matrix, n_vgrids, sigma_matrix, prior, matrix_components=2, log_sigma=True):
        
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

        for v in tqdm.tqdm(range (model_matrix.shape[0])):
        
            for i in range(observation_matrix.shape[0]):
        
                variance_flat=sigma_matrix[i].flatten()
        
                print (variance_flat.shape)
        
                mask_finite=np.isfinite(observation_matrix[i])
                mask_finite_var=np.isfinite(variance_flat)
        
        
                if matrix_components == 1:
                    fit_model_mask=np.array([model_matrix[v][i][mask_finite&mask_finite_var]])
                elif matrix_components == 2:
                    fit_model_mask=np.array([model_matrix[v][0,i][mask_finite&mask_finite_var], model_matrix[v][1, i][mask_finite&mask_finite_var]])
                else:
                    print ("The first axis of your model matrix refers to the number of components in your model. It is supposed to be either 1 or 2 in EXOCRIRES v1.0.")
        
                    continue
                    
                output=self.likelihood_calculator(d=observation_matrix[i][mask_finite&mask_finite_var], M=fit_model_mask.T, \
                                                Sigma_0_flat=variance_flat[mask_finite_var&mask_finite], log_sigma_0=log_sigma, gamma=2, prior_psi=prior)
        
                
                log_likelihood_mapp[i][v]=output
        
                fit_model_mask = None
            

        return (log_likelihood_mapp)
                

