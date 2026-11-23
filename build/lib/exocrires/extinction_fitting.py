# Cooking the input spectra

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import fftpack,signal
import matplotlib
from matplotlib.ticker import ScalarFormatter
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize, ZScaleInterval)
import scipy
import os
import emcee
import tqdm
import corner

from exocrires import info
import PyMieScatt as scatt

class  extinction_fitting:

    def __init__(self, wavelength_sed, flux_sed, flux_obs, flux_obs_err, flux_int=None, flux_ext=None, resolution=None, extinction_law=None, radius_range=None, av_range=None, band_center=None, beta=None):

        self.wavelength=wavelength_sed
        self.flux=flux_sed

        self.law=extinction_law
        self.radius_range=radius_range
        self.av_range=av_range
        self.band_center=band_center
        self.beta=beta

        self.flux_obs=flux_obs
        self.flux_obs_err=flux_obs_err

        self.bandv_center=550 #nm


    def Flux_integral(self, band_border, transmission):

        flux_int=np.zeros(shape=(len(band_border),))

        #dist_ratio=((params[0]*r_j)/dis)

        for i in range(len(band_border)):

            mask=np.array((self.wavelength>band_border[i][0])&(self.wavelength<band_border[i][1]))

            data_cut=np.array([self.wavelength[mask], self.flux[mask]])

            transmi=np.interp(data_cut[0], transmission[i][0], transmission[i][1])
            flux_int[i] = np.trapz(data_cut[1]*transmi, data_cut[0])/np.trapz(transmi, data_cut[0])

        self.flux_int=flux_int
        return (flux_int)


    #------------------


    #--------------
    def MCMC(self, n_walkers, n_steps, n_params, name_params, lower_bounds, upper_bounds, dis):

        x=self.flux_int
        y=self.flux_obs
        y_err=self.flux_obs_err
        r_j = 6.9950e9 #cm

        #------------------

        # Define likelihood and prior functions

        def log_likelihood(params, x, y, y_err):

            #params=[radius, extinction]

            #x: integral flux

            #Pay attention: The calculation should not envolve zero-point flux. The observed flux (y) you retrieved from exocrires.analysis.mag_convert should NOT be corrected by zero-point flux of the telescope system

            dist_ratio=(params[0]*r_j)/dis
            #--ISM Extinction
            if extinction_law == 'ISM':

                dist_ratio=((params[0]*r_j)/dis)

                a_lam=params[1]*((band_center/bandv_center)**(beta))


                f_x=0.5*np.square(dist_ratio)*self.f_int*(10**(-params[1]/2.5))


            chi_2=np.square((y-f_x)/y_err)

            N=len(x)

            N_half=N/2

            likelihood = (1/(2*np.pi)**N_half)*(1/np.prod(y_err))*np.e**(-0.5*np.sum(chi_2))
            log_li = -0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(y_err**2) + chi_2)

            return (log_li, likelihood, f_x, chi_2)

        def log_prior(params, l_bounds=lower_bounds, u_bounds=upper_bounds):

            # prior function
            if all(lower <= param <= upper for param, lower, upper in zip(params, lower_bounds, upper_bounds)):

                return 0.0

            else:
                return (-np.inf)



            # Define the log posterior (log likelihood + log prior)
        def log_posterior(params, x, y, y_err):

            lp = log_prior(params)

            if not np.isfinite(lp):

                return (-np.inf)

            return (lp + log_likelihood(params, x, y, y_err)[0])

        #-----------

        # Initialize walkers with random starting points
        initial_positions = np.random.rand(n_walkers, n_params)

        #Set up the array for best-fitting parameters and other arguments
        best_fitting_params_ex=np.zeros((n_params+1, ))


        # Set up the sampler
        arguments=(self.flux_int, self.flux_obs, self.flux_obs_err) #x,y,y_err

        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior, args=arguments)

        # Run the MCMC sampler
        s=sampler.run_mcmc(initial_positions, n_steps, progress=True)

        # Get the samples from the chain
        samples = sampler.get_chain()

        # Calculate parameter estimates (e.g., mean, median, etc.)
        parameter_estimates = np.mean(samples, axis=0)

        # Save the flat samples
        flat_samples = sampler.get_chain(discard=50, thin=20, flat=True)
        print(flat_samples.shape)

        #Visulize the results
        fig = corner.corner(flat_samples, labels=['radius','Av'])
        #plt.savefig('./MCMC/3004/%s_%s_%s_3bands.png'%(parameter_se[i][0], parameter_se[i][1], n_params))
        plt.show()

        # Identify best-fitting parameters based on mean-error minimization
        log_posterior_data = np.array([log_posterior(sample, arguments[0], arguments[1], arguments[2]) for sample in flat_samples])

        best_fitting_index = np.argmax(log_posterior_data) # criterion for selecting the best parameters

        best_fitting_params_ex[0:n_params] = flat_samples[best_fitting_index]
        best_fitting_params_ex[n_params] = np.max(log_posterior_data)

        return (flat_samples, best_fitting_params_ex)
