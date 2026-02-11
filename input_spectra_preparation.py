import astropy
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.linalg import solve

import scipy.ndimage
import scipy.constants

from scipy import interpolate
from astropy.io import fits
from PyAstronomy import pyasl
from exotic_ld import StellarLimbDarkening

import matplotlib.pyplot as plt
import tqdm

import glob




import os

class Spectra_Preparator:
    '''
    Class to prepare and process stellar spectra.
    '''
    def __init__(self, path, T, G, Z, folder_name=None):
        """
        Initialize the Spectra_Preparator class.
        T,G,Z: Stellar or planetary parameters (Temperature, Gravity, Metallicity).
        """
        self.path = path
        self.folder_name = folder_name
        self.path_a = os.path.join(path, folder_name)
        self.T = T
        self.G = G
        self.Z = Z
        self.ld_coefficients = None

    def generate_synthetic_stellar_models(self, model_spec, n_mus, factor):
        """
        Generate synthetic stellar models based on a given spectrum.
        model_spec: 1D numpy array representing the model spectrum.
        n_mus: Number of angles (mu) to consider.
        factor: Scaling factor for the intensity.
        """
        # Generate I(lambda, mu).
        n_wvs = len(model_spec)
        mus = np.linspace(0, 1, n_mus)
        stellar_intensity = []

        for mu in mus:
            stellar_intensity.append(model_spec*factor)

        return mus, np.array(stellar_intensity).T

    def compute_limb_darkening(self, wavelength_range, throughput_files, Mode='phoenix', flux=None, wave=None):
        """
        Compute limb darkening coefficients for different bands.
        wavelength_range: Range of wavelengths to consider.
        throughput_files: List of throughput files for each band.
        Mode: Mode of limb darkening computation ('phoenix' or 'custom').
        flux: Flux array for custom mode.
        wave: Wavelength array for custom mode.
        """
        ld_coeffs = np.zeros((len(wavelength_range), 1))

        for idx, (lam_range, throughput_file) in enumerate(zip(wavelength_range, throughput_files)):
            throughput = np.loadtxt(throughput_file)
            wave_throughput, frac_throughput = throughput[:, 0], throughput[:, 1]

            if Mode == 'phoenix':
                sld = StellarLimbDarkening(
                    ld_model=Mode,
                    ld_data_path=os.path.join(self.path, 'ld_data'),
                    M_H=self.Z,
                    Teff=self.T,
                    logg=self.G
                )

                cs = sld.compute_linear_ld_coeffs(
                    mode='custom',
                    wavelength_range=[lam_range[0][0]*10, lam_range[-1][-1]*10],
                    custom_wavelengths=wave_throughput,
                    custom_throughput=frac_throughput
                )

            elif Mode =='custom':
                mus, intensities = self.generate_synthetic_stellar_models(flux, 70, factor=1e-8)
                
                sld=StellarLimbDarkening(
                    ld_model=Mode, 
                    ld_data_path=os.path.join(self.path, 'ld_data'), \
                    custom_wavelengths=wave, 
                    custom_stellar_model=intensities, 
                    custom_mus=mus
                )
                


                cs=sld.compute_linear_ld_coeffs(
                    mode='custom', wavelength_range=[lam_range[0][0]*10, lam_range[-1][-1]*10],   
                    custom_wavelengths=wave_throughput, 
                    custom_throughput=frac_throughput
                )




            ld_coeffs[idx] = np.float16(cs)

        self.ld_coefficients = ld_coeffs
        return ld_coeffs

    def load_phoenix_spectrum(self):
        """
        Load high resolution PHOENIX model spectrum.
        """
        flux = fits.open(os.path.join(
            self.path_a,
            f"lte0{self.T}-{self.G}0{self.Z}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        ))[0].data

        wave = fits.open(os.path.join(
            self.path_a,
            "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
        ))[0].data

        return wave, flux

    def gaussian_broaden(self, wavelength, flux, R):
        """
        Apply Gaussian broadening to the spectrum for LSF convolution.
        R: Spectral resolution.
        """
        log_wavelength = np.log(wavelength)
        fwhm_log = 1 / R
        sigma_log = fwhm_log / 2.355
        broadened_flux = scipy.ndimage.gaussian_filter1d(
            flux,
            sigma=sigma_log / (log_wavelength[1] - log_wavelength[0])
        )
        return broadened_flux

    def resolution_sample(self, ave_R, wavelength):
        lam_range = np.max(wavelength) - np.min(wavelength)
        lam_mid = (np.max(wavelength) + np.min(wavelength)) / 2
        N = ave_R * (lam_range / lam_mid) + 1
        return int(N)

    def apply_broadening(self, wave, flux, lam_range, ld_coeff, R=1e5, rotation=5e0, reu=500, step_size=1e-12, interpolation='spline', wave_template=None):
        """
        Calculate rotational broadening and Apply all broadening effects to the spectrum.

        Variables:
        wave: Wavelength array of the spectrum.
        flux: Flux array of the spectrum.
        lam_range: Wavelength range for cutting the spectrum.
        ld_coeff: Limb darkening coefficients.
        R: Spectral resolution.
        rotation: Stellar rotation velocity.
        reu: Radial velocity shift.
        step_size: Step size for interpolation. It can be string {2x, mean} or float value.
        interpolation: Interpolation method ('linear' or 'spline').
        wave_template: Default as 'None'. It is necessary when using 'spline' interpolation. We suggest to use the wavelength solution from telluric model calculated from ETC.
        """

        if (wave[1]-wave[0]) != (wave[-1]-wave[-2]):

            print ('Sample grid is not equal. Use pyasl.equidistantInterpolation to produce equidistantly sampled data. ')

            # Perform equidistant interpolation
            if interpolation =='spline':
                tck = scipy.interpolate.splrep(wave, flux, s=0.0)
                #wave_new = np.arange(np.min(wave), np.max(wave), step_size)
                if wave_template is None:
                    wave_new, _ = pyasl.equidistantInterpolation(wave, flux, step_size)
                else:
                    wave_new = wave_template

                flux_new = scipy.interpolate.splev(wave_new, tck, der=0, ext=2)

            elif interpolation =='linear':
                wave_new, flux_new = pyasl.equidistantInterpolation(wave, flux, step_size)

    
        mask =  ( (wave_new >= (lam_range[0][0]-reu)*1e-9) & (wave_new <= (lam_range[-1][-1]+reu)*1e-9) )
        rflux = pyasl.fastRotBroad(
            wvl=wave_new[mask],
            flux=flux_new[mask],
            epsilon=ld_coeff,
            vsini=rotation
        )
        broadened_flux = self.gaussian_broaden(wave_new[mask], rflux, R)

       
        if wave_template is not None:
            wave_grid = wave_template

        else:
            # Resample to desired resolution is the dataset tends to be too large from oversampling
            N = self.resolution_sample(R*2, wave_new[mask])
            wave_grid = np.linspace(np.min(wave_new[mask]), np.max(wave_new[mask]), N)

        if interpolation == 'linear':
            inter_flux = np.interp(wave_grid, wave_new[mask], broadened_flux)

        elif interpolation == 'spline':
            tck = scipy.interpolate.splrep(wave_new[mask], broadened_flux, s=0)
            inter_flux = scipy.interpolate.splev(wave_grid, tck)

        return wave_grid, inter_flux


    def radial_velocity_shift(self, wave, flux, velocity):
        """
        Apply radial velocity shift in km/s.
        """
        shifted_wave = wave * (1 + (velocity * 1e3 / scipy.constants.c))
        spec_shift = np.array([shifted_wave, flux]).T
        return spec_shift
    
    def spectrum_cutter(self, lam_range, wave, flux, saving_file=True, saving_path = None, filenames = None, ignore_detector=True, Wave_solution=None):
        '''
        Cut the spectrum into different orders or detectors.
        lam_range: Wavelength range for cutting the spectrum.
        wave: Wavelength array of the spectrum.
        flux: Flux array of the spectrum.
        saving_file: Boolean indicating whether to save the cut spectra.
        saving_path: Path to save the cut spectra.
        filenames: List of filenames for the cut spectra.
        ignore_detector: Boolean indicating whether to ignore detector division.
        Wave_solution: Wavelength solution for the spectrum.
        '''
        
        if ignore_detector == True:
            print ('The division into 3 detectors is ignored. The spectrum is only cut into different spectral orders.')
            n=0
            
            for i in lam_range:

                mask = ( ( wave >= i[0]*1e-9) & (wave <= i[1]*1e-9) )

                wave_cut = wave[mask]
                flux_cut = flux[mask]

                if saving_file == True:
                
                    filename = saving_path + filenames[n]

                    self.save_spectrum(filename, wave_cut, flux_cut)

                    n+=1
            
            print ('All spectra are saved in %s'%saving_path)

        if ignore_detector == False:

            spec_matrix = self.model_matrix_intepolater(wave, flux, Wave_solution)

            if saving_file == True:
                
                filename = '%s'%saving_path + '%s'%filenames

                np.save(filename, spec_matrix)

            print ('The spectral matrix is saved in %s with the shape of %s'%(saving_path, spec_matrix.shape))

            return (spec_matrix)


    def model_matrix_intepolater (self, wave_model, flux_model, wave_solution, interpolation='spline', ignore_detector=True, saving_file = True, saving_path = None, filename = None):
        """
        Interpolate the model spectrum onto the desired wavelength solution.
        wave_solution: Wavelength solution for the spectrum, usually from the transmission model. The shape must be (n_orders, n_detectors, n_pixels, 2)
        """
        #check up the shape of wave_model and wave_solution
        print("wave_model shape:", wave_model.shape)
        print("wave_solution shape:", wave_solution.shape)

        spec_interp_matrix = np.zeros( shape=wave_solution.shape )

        if ignore_detector == True:

            for order in range(wave_solution.shape[0]):

                wave_cut = wave_solution[order, :, 0]
                flux_cut = wave_solution[order, :, 1]

                model_mask = np.array((wave_model>=wave_cut.min())&(wave_model<=wave_cut.max()))

                if interpolation == 'linear':
                    spec_interp = np.interp(wave_cut, wave_model[model_mask], flux_model[model_mask])

                elif interpolation == 'spline':
                    tck = scipy.interpolate.splrep(wave_model[model_mask], flux_model[model_mask], s=0)
                    spec_interp = scipy.interpolate.splev(wave_cut, tck)

                spec_interp_matrix[order, :, 0]=wave_solution[order, :, 0]
                spec_interp_matrix[order, :, 1]=spec_interp

        else:
            for order in range(wave_solution.shape[0]):
                for det in range(wave_solution.shape[1]):
                    spec_interp_matrix[order, det, 0] = wave_solution[order, det, 0]
                    spec_interp_matrix[order, det, 1] = wave_solution[order, det, 1]

                    model_mask = np.array((wave_model >= wave_solution[order, det, 0].min()) & (wave_model <= wave_solution[order, det, 0].max()))

                    if interpolation == 'linear':
                        spec_interp = np.interp(wave_solution[order, det, 0], wave_model[model_mask], flux_model[model_mask])
                    elif interpolation == 'spline':
                        tck = scipy.interpolate.splrep(wave_model[model_mask], flux_model[model_mask], s=0)
                        spec_interp = scipy.interpolate.splev(wave_solution[order, det, 0], tck)

                    spec_interp_matrix[order, det, 0] = wave_solution[order, det, 0]
                    spec_interp_matrix[order, det, 1] = spec_interp
        
        if saving_file == True:
            filename = '%s'%saving_path + '%s'%filename
            np.save(filename, spec_interp_matrix)
        
        return (spec_interp_matrix)
        
    
    #-----Lorenzo's functions for convolution in velocity space-----#
    def convolve_model(self, wave_model, F, vsini, instrument_resolving_power):

        # Taken from Mike Line:
        #
        # Explanation: calculate the delta_v step for every wavelength bin. delta_lambda/lambda = delta_v/c. Then:
        #
        # delta_lambda = l[1:] - l[0:-1];
        # lambda_center_of_bin = (l[1:] + l[0:-1])/2
        #
        # Invert for delta_v, and you get the following. Mike calculates the average step, I pass the array and then check inside convolve_rotins_rv_space that the delta_v step is constant or nearly so.

        v_step = 2.0*(wave_model[1:]-wave_model[0:-1])/(wave_model[1:]+wave_model[0:-1])*2.998E5

        if not np.all(np.isclose(v_step, v_step[0])):
            raise IOError('Make sure that your velocity grid is constantly spaced (grid equally spaced in log_lambda)')

        F_conv = self.convolve_rotins_rv_space(None, F, v_step=v_step[0], vsini=vsini, LSF_FWHM=c_km_s/instrument_resolving_power, epsilon=0.)

        return F_conv


    def rotational_broadening_kernel(self, deltav, vsini, epsilon=0.):
        """Translated from IDL.

        https://idlastro.gsfc.nasa.gov/ftp/pro/astro/lsf_rotate.pro
        """

        e1 = 2.*(1. - epsilon)
        e2 = np.pi*epsilon/2.
        e3 = np.pi*(1. - epsilon/3.)

        n_points = math.ceil(2*vsini/deltav)
        if n_points%2 == 0.:
            n_points += 1

        nwid = int(n_points/2)
        x = (np.arange(n_points) - nwid)
        x = x*deltav/vsini
        velocity_grid = x*vsini
        # if arg_present(velgrid) then velgrid = x*vsini
        x1 = abs(1.0 - x**2.)

        rot_kernel = (e1*np.sqrt(x1) + e2*x1)/e3

        return velocity_grid, rot_kernel
    
    def convolve_rotins_rv_space(self, velocities, y,vsini=0., LSF_FWHM=0., epsilon=0., v_step=None):
        """

        Input:

            vsini: Rotational velocity

            LSF_FWHM: The full width half maximum of the instrumental profile in km / s

            epsilon: Linear limb darkening coefficient

            vstep: [optional] Use a fixed velocity step rather than deriving it (useful if you have a grid in lambda)

        """

        if not (vsini or LSF_FWHM):
            raise IOError('What are you trying to convolve? Boradening is set to zero.')

        if not v_step:
            v_step = np.diff(velocities)
            if not np.all(np.isclose(v_step, v_step[0])):
                raise IOError('Make sure that your velocity grid is constantly spaced')
            v_step = v_step[0]

        convolved_CCF = np.zeros(len(y))

        if vsini != 0.:
            velocity_grid_kernel, rot_kernel = self.rotational_broadening_kernel(v_step, vsini, epsilon)
            convolved_CCF = astropy.convolution.convolve(y, rot_kernel, boundary='extend')

        if LSF_FWHM != 0.:

            stddev_km_sec = LSF_FWHM/2.355
            stddev_pixels = stddev_km_sec/v_step
            g_kernel = astropy.convolution.Gaussian1DKernel(stddev_pixels)
            # velocity_grid_kernel, g_kernel = gaussian_broadening_kernel(v_step[0], LSF_FWHM)
            convolved_CCF = astropy.convolution.convolve(convolved_CCF, g_kernel, boundary='extend')

    def save_spectrum(self, filename, wave, flux):
        """
        Save the spectrum to a file.
        """
        spec = np.array([wave, flux])
        np.savetxt(filename, spec)

        return (spec)