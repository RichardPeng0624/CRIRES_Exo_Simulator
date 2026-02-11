import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import fftpack,signal
import matplotlib
from matplotlib.ticker import ScalarFormatter
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize, ZScaleInterval)
import scipy
import json
import subprocess
import os


class spectra_2d:

    #------------------

    def __init__ (self, path_etc_local, path_etc_input, path_input_modi, \
                  path_etc_output, aperture_size, date, order_number=[], order_window=[], \
                  target = None, model_name=None, planet_p=None):

        #doc strings:

        '''

        target: set up the target

        path_etc_local: the local path of the etc script linked to website service

        path_etc_input: the path of input data for etc

        path_input_modi: the path to save the input data after modification in codes

        path_etc_output: the path to save the output data from etc

        aperture size: The maximum size you would like to use for the extraction aperture. ATTENZIONE: the ODD number is  suggested for gradually increasing the aperture size and making difference.

        '''

        if target not in ['star', 'planet', 'disk']:
            print ('Target not valid, please choose one from "star" or "planet" or "disk" ') #The target can only be either star or planet

        self.target = target #set up the target
        self.path_etc_local=path_etc_local #the local path of the etc script linked to website service
        self.path_input = path_etc_input #the path of input data for etc
        self.path_input_modi = path_input_modi #the path to save the input data after modification in codes

        self.path_output = path_etc_output #the path to save the output data from etc

        if os.path.exists(self.path_output) == False:  #create the paths of output files if they don't exist
            subprocess.call(['mkdir', self.path_output])

        self.model_name = model_name
        self.date=date

        self.order_num = order_number
        self.order_wave = order_window

        self.planet_p = planet_p #planet position

        if aperture_size%2 == 0: #we need the odd number for increasing the aperture size later
            print ("Warning: It's better to use an odd number, the final size is better to be %s"%(aperture_size-1))

        self.aperture_size = aperture_size
        self.aperture_list = np.arange(1, self.aperture_size+1, 2).tolist() # for example: 1, 3, 5 ...35, we create the list with ascending numbers for the aperture size

    #------------------
    def output_json (self, star_template=None):

        '''
        # The function aimes to create and save the output data from etc in .json format automatically
        # With the list created in the initial function, we are able to produce a set of output data with ascending apeture sizes, e.g. from 1, 3, 5...to 35
        # Later in the next few functions, we gonna make difference with these output data to retrieve the 2-D spectra.
        # star_template: If you want to use your own stellar template, please pass the path in the arugment here
        '''

        '''
        --last modified on 4th, Feburary, 2026 to be compatible with the new ETC v117.2.0--
        '''

        with open(self.path_input, 'r') as f:
            data = json.load(f)
            # Modify the contents of the JSON file
            for i in self.aperture_list:

                data['seeing']['aperturepix']= float(i)

                # Update the modified JSON file to disk
                with open(self.path_input_modi, 'w') as f:
                    json.dump(data, f)

                subprocess.call(['mkdir', self.path_output+'ascii_%s/'%i])

                if self.target == 'star':

                    if star_template==None:

                        subprocess.call(['python', self.path_etc_local, 'CRIRES', self.path_input_modi, \
                             '-o', self.path_output+'ascii_%s/'%i+'output_%s_%s_AO.json'%(self.date, i)])

                    else:

                        subprocess.call(['python', self.path_etc_local, 'CRIRES', self.path_input_modi, \
                         '-u' , star_template, '-o', self.path_output+'ascii_%s/'%i+'output_%s_%s_AO.json'%(self.date, i)])


                elif self.target == 'planet':
                    subprocess.call(['python', self.path_etc_local, 'CRIRES', self.path_input_modi, \
                         '-u' , self.model_name, '-o', self.path_output+'ascii_%s/'%i+'output_p_%s_%s_AO.json'%(self.date, i)])

        print ('the ETC calculation is done, the output file is saved at %s'%self.path_output)

    #------------------

    def json2ascii (self, path_json2ascii):

        '''
        # This function works on converting the format froom json to ascii assoicated with a script provided by ETC page of CRIRES+
        # ATTENZIONE: NOT ALL OUTPUT DATA CAN BE SUCCESSFULLY CONVERTED
        # Please check carefully for the .ascii files and if you can, just use .json
        '''
        print ('Conversion from json to ascii starts')

        if self.target == "star":

            for i in self.aperture_list:
                subprocess.call(['cp', '-r', path_json2ascii, self.path_output+'ascii_%s/'%i])
                subprocess.call(['python', self.path_output+'ascii_%s/'%i+'etc_json2ascii.py',\
                                 self.path_output+'ascii_%s/'%i+'output_%s_%s_AO.json'%(self.date, i)])

        if self.target == 'planet':
            for i in self.aperture_list:
                subprocess.call(['cp', '-r', path_json2ascii, self.path_output+'ascii_%s/'%i])
                subprocess.call(['python', self.path_output+'ascii_%s/'%i+'etc_json2ascii.py',\
                                 self.path_output+'ascii_%s/'%i+'output_p_%s_%s_AO.json'%(self.date, i)])

    #------------------

    def ascii2txt (self, output, order_number=None):

        '''
        # This functions works on converting .ascii files into .txt files
        # Order_number: the order numbers of the detector. The defalut setting follows the number from __init__

        '''

        if order_number == None:
            order_number=self.order_num

        for i in self.aperture_list:

            for j in [1,2,3]:
                for x in order_number:
                    for m in output:
                        try:
                            if self.target == 'star':
                                file=self.path_output+'ascii_%s/'%i+\
                                'output_%s_%s_AO.json_order:%s_det:%s_%s.ascii'%(self.date, i, x, j, m)
                            elif self.target =='planet':
                                file=self.path_output+'ascii_%s/'%i+\
                                'output_p_%s_%s_AO.json_order:%s_det:%s_%s.ascii'%(self.date, i, x, j, m)
                            dat=[]
                            with open(file, 'r') as f:
                                lines = f.readlines()
                                dat += [line.split() for line in lines[2:]]


                            with open(self.path_output+'ascii_%s/'%i+'converted_data_%s_%s_%s.txt'%(x,m,j), 'w') as f:
                                # Write the header line
                                if m == 'psf':
                                    f.write("angle (arcsec)\tPSF (norm)\n")
                                elif m == 'snr_snr':
                                    f.write('wavelength (m)\tsnr\n')
                                elif m == 'sed_target':
                                    f.write('wavelength (m)\tflux (Jsm-2m)\n')
                                else:
                                    f.write("wavelength (m)\tcounts\n")
                                # Write the data
                                for line in dat:
                                    f.write("\t".join(line) + "\n")


                        except FileNotFoundError:
                            print('File Not Found: %s-%s-%s-%s'%(i,x,j,m))
                            continue

                    print('txt_done_%s_%s_%s'%(i,x,j))

    #------------------

    def signal(self, detectors, contribution=None, nor=None, focal_plane=None, Signal=None, SED=None):

        '''
        This is the core function of this class.
        The aim of this function is to re-distribute the signal in a 2-D plane
        ------
        #focal_plane: the focal plane size in pixels. e.g. (2046,2046). The default value is from __init__
        #detectors: number of detectors
        #contribution: sky, target or total
        #nor: normalization the map or not, the option is TRUE or FALSE; The default setting is normalization
        '''

        self.focal_plane = np.zeros(focal_plane)
        self.detectors = detectors

        order_number = self.order_num
        order_window = self.order_wave

        if contribution not in ['sky', 'target', 'tot']:
            print('Stupid boy, you need to choose one from [sky, target, tot]')


        inter = np.arange(0, int(np.ceil(self.aperture_size/2)),1)
        name = self.aperture_list


        #build the dict for the data with ascending aperture sizes
        #the dict follows the structure['aper': serial number (0,1,2,3,...17)['wavelength(nm)','counts'], 'name': aperture size (1,3,5...35)]
        #------------------------------signal------------------------------------------------------------
        if Signal!=False:

            dat_tot = {'aper': inter.tolist(), 'name': name}

            for n in inter.tolist():
                i = 2*n+1
                dat_tot['aper'][n] = {'wavelength(nm)':[],'counts':[]}

                for nu in range(1, detectors+1):
                    for order in order_number:

                        data_sig=pd.read_table(self.path_output+'ascii_%s/'%i+'converted_data_%s_signals_obs%s_%s.txt'%(order,contribution, nu))
                        dat_tot['aper'][n]['wavelength(nm)'].append(data_sig['wavelength (m)'].values*1e9)
                        dat_tot['aper'][n]['counts'].append(data_sig['counts'].values)




            #set up the dataframe consisting of the counts for each row . Let's start from the central pixel (with aper size =1)
            wave_count_0=pd.DataFrame(data=dat_tot['aper'][0])

            #make difference for each two aperture sizes and write the data into the dataframe wave_count_0

            for i in inter[1:]:

                wave_count_i=pd.DataFrame(data=dat_tot['aper'][i])
                wave_count_i_1=pd.DataFrame(data=dat_tot['aper'][i-1])
                if np.where((wave_count_i['wavelength(nm)']-wave_count_i_1['wavelength(nm)']).all==0, True, False) == False:
                    dat_diff=(wave_count_i['counts']-wave_count_i_1['counts'])/2
                else:
                    print('Wrong result! The wavelength range is not matched')
                wave_count_0['dat_diff_%s'%i]=dat_diff

            #organize wave_count_0 for ploting: rename, explode, and sort the values by ascending wavelength

            wave_count_0=wave_count_0.rename(columns={'counts':'dat_diff_0'})
            name_list=wave_count_0.columns.to_list()
            wave_count_ex=wave_count_0.explode(name_list).reset_index()
            wave_count_sort=wave_count_ex.sort_values(by='wavelength(nm)').reset_index()

        #-----------------------------------SED----------------------------------------------------------

        if SED == True:

            dat_tot_sed = {'aper': inter.tolist(), 'name': name}

            for n in inter.tolist():
                i = 2*n+1
                dat_tot_sed['aper'][n] = {'wavelength(nm)':[],'flux (Jsm-2m)':[]}

                for nu in range(1, detectors+1):
                    for order in order_number:

                        data_sig=pd.read_table(self.path_output+'ascii_%s/'%i+'converted_data_%s_sed_%s_%s.txt'%(order,contribution, nu))
                        dat_tot_sed['aper'][n]['wavelength(nm)'].append(data_sig['wavelength (m)'].values*1e9)

                        if contribution == 'sky':
                            dat_tot_sed['aper'][n]['flux (Jsm-2m)'].append(data_sig['counts'].values)
                        else:
                            dat_tot_sed['aper'][n]['flux (Jsm-2m)'].append(data_sig['flux (Jsm-2m)'].values)



            #set up the dataframe consisting of the counts for each row . Let's start from the central pixel (with aper size =1)
            wave_count_sed_0=pd.DataFrame(data=dat_tot_sed['aper'][0])

            #make difference for each two aperture sizes and write the data into the dataframe wave_count_0

            for i in inter[1:]:

                wave_count_sed_i=pd.DataFrame(data=dat_tot_sed['aper'][i])
                wave_count_sed_i_1=pd.DataFrame(data=dat_tot_sed['aper'][i-1])
                if np.where((wave_count_sed_i['wavelength(nm)']-wave_count_sed_i_1['wavelength(nm)']).all==0, True, False) == False:
                    dat_diff=(wave_count_sed_i['flux (Jsm-2m)']-wave_count_sed_i_1['flux (Jsm-2m)'])/2
                else:
                    print('Wrong result! The wavelength range is not matched')
                wave_count_sed_0['dat_diff_%s'%i]=dat_diff

            #organize wave_count_0 for ploting: rename, explode, and sort the values by ascending wavelength

            wave_count_sed_0=wave_count_sed_0.rename(columns={'flux (Jsm-2m)':'dat_diff_0'})
            name_list=wave_count_sed_0.columns.to_list()
            wave_count_sed_ex=wave_count_sed_0.explode(name_list).reset_index()
            wave_count_sed_sort=wave_count_sed_ex.sort_values(by='wavelength(nm)').reset_index()
        if Signal!=False:

            if SED == True:

                return (wave_count_sort, wave_count_sed_sort)

            else:

                return (wave_count_sort)

        else:
            return (wave_count_sed_sort)

    #------------------

    def plot_signal(self, wave_count_sort, max_percentile, save_path, interv, minimum=None, Nor=None, fontsize=None):
        '''
        This is used to virualize the signal as a 2-D spectra.

        # wave_count_sort: the data set of the focal plane
        # max_percentile: the maximum percentile for the plot
        # save_path: the path to save your plots
        # minimum: minimum percentile for the plot
        # Nor: if None, normalization is following the func: ImageNormalize(focal_plane, interval= interv, vmax=ceil, vmin=minimum)
        # fontsize: font size of the plots

        '''

        #set up the fontsize
        if fontsize==None:
            font = {'size': 4}
        else:
            font = {'size': fontsize}

        plt.rcParams.update({'font.size': font['size']})

        #plot data with imshow
        plt.ion() #enable the interactive mode
        row = len(self.order_num)
        fig, axs=plt.subplots(row, 1, dpi=500, sharex=True, sharey=True)
        focal_plane = self.focal_plane

        for n_order in range(row):

            one_order=wave_count_sort.loc[(wave_count_sort['wavelength(nm)']<=self.order_wave[n_order][1])&(wave_count_sort['wavelength(nm)']>=self.order_wave[n_order][0])]

            central_pix=int(np.ceil(focal_plane.shape[0]/2))

            pixel_posi=len(self.aperture_list)
            for x in range(0,pixel_posi):
                loc_up=central_pix+x
                loc_be=central_pix-x
                self.focal_plane[loc_up,0:len(one_order)]=one_order['dat_diff_%s'%x]
                self.focal_plane[loc_be,0:len(one_order)]=one_order['dat_diff_%s'%x]

            ff=axs[n_order]

            ceil= np.percentile(focal_plane, max_percentile)
            if minimum == None:
                minimum=np.min(focal_plane)

            nor = ImageNormalize(focal_plane, interval= interv, vmax=ceil, vmin=minimum)

            max_order = np.max(self.order_num)


            ff.set_title('order%s: %s nm - %s nm'%(max_order-n_order,self.order_wave[n_order][0],self.order_wave[n_order][1]),fontsize=5)
            if Nor == None or Nor == True:
                ff.imshow(focal_plane,norm = nor ,cmap='Greys_r')
                fig.colorbar(ff.imshow(focal_plane, norm=nor, cmap='Greys_r'), ax=axs[n_order], shrink=0.5, aspect=60, \
                         location='bottom',pad=0.2)
            elif Nor == False:
                ff.imshow(focal_plane, vmin=minimum, vmax=ceil, cmap='Greys_r')
                fig.colorbar(ff.imshow(focal_plane, vmin=minimum, vmax=ceil, cmap='Greys_r'), ax=axs[n_order], shrink=0.3, aspect=50, \
                         location='bottom',pad=0.2)
            else:
                print('Stupid sweet, you just passed a wrong value to Nor. It should be on from True, Flase or None')

            ff.set_aspect(8)
            ff.axis('off')
            '''
            if "plot_combination" == True:
                ff.hlines(y=1024+self.planet_p, xmin=0, xmax=self.focal_plane.shape[1], ls='--',colors='gray', linewidth=0.8)
            '''
        plt.subplots_adjust(hspace=1)
        plt.ylim(loc_be-1,loc_up+1)
        plt.suptitle('focal plane %s \n$e^-$/pix/exposure'%str(focal_plane.shape),x=0.5,fontsize=9)

        plt.savefig(save_path)
        plt.close()

    #------------------

    def combine(self, data_star, data_p,\
                data_disk=None, disk_structure=None, disk_separation=None, \
                plot_combination=None, d=None, star_posi=None, sky='Uniform', noise=True, data_sky=None,\
                ron=None, dark=None, NDIT=None, plane=None, ceil_percentage=None,\
                plot_save_path=None):

        '''
        This function works for combining all signals together in one plot to to simulate the observation.

        # data_star, data_p, data_disk: the dataset of the whole focal plane for star, planet and disk
        # disk structure: ['two-side ring', 'one-side ring', 'veiling'] to determine the disk contribution along the spatial axis
        # disk separation: in pixel. Used to determine the disk contribution along the spatial axis
        # plot_combination: if plot the combination signal map or not. The default is not plotting
        # d: The projected separation between the planet and the star
        # star_posi: The position of the star on the spatial axis of the detector plane (could be used for nodding mode). The default position is the center of the spatial axis.
        # sky: ['Uniform', 'Varation', False] if 'Uniform', consider the uniform sky background emission for the whole detector plane. The default is 'Uniform'.
        # noise: [True, False] if true, consider the noise terms. The defalut is True. The noise is Poisson noise.
        # data_sky: The path the sky data.
        # ron: mean read-out-noise (lambda in a Poisson distribution)
        # dark: mean value of dark counts
        # plane: the size of focal plane, (interger, interger). The defalut setting is to use the one from __init__
        # plot_save_path: the path to save plots
        '''

        font = {'size': 4}
        plt.rcParams.update({'font.size': font['size']})

        if noise not in [True, False]:

            print('Stupid, you just passed a nonsense value to noise. It should be one from [None, True, False]')

        if sky not in [False, 'Uniform', 'Variation']:

            print('Stupid, you just passed a nonsense value to sky contribution. It should be one from [None, True, False]')

        if self.planet_p == None:
            self.planet_p = d

        elif d!=0 and self.planet_p != None:
            print ('Stupid! you already choose one position in the initial function. I will not change the value!')
            d = self.planet_p

        if plane != None:
            self.focal_plane = np.zeros(shape=plane)

        if star_posi == None:

            central_pix=int(np.ceil(self.focal_plane.shape[0]/2))

        else :
            central_pix = star_posi


        row=len(self.order_num)
        pixel_posi=len(self.aperture_list)

        if NDIT == None:
            signal_tot = np.zeros(shape=(row, self.focal_plane.shape[0], self.focal_plane.shape[1]))

        else:
            signal_tot = np.zeros(shape=(row, NDIT, self.focal_plane.shape[0], self.focal_plane.shape[1]))

        signal_stack = np.zeros(shape=(row, self.focal_plane.shape[0], self.focal_plane.shape[1]))

        #if noise==True:
        #exposure_series=np.zeros(shape=(N, row, focal_plane.shape[0], focal_plane.shape[1]))

        for n_order in range(row):

            if NDIT == None:
                focal_plane = signal_tot[n_order]
            else:
                focal_plane = np.zeros(shape=plane)

            cons_plane = self.focal_plane
            cons_nnoise_plane = self.focal_plane

            order_w=self.order_wave

            one_order_s=data_star.loc[(data_star['wavelength(nm)']<=order_w[n_order][1])&(data_star['wavelength(nm)']>=order_w[n_order][0])]

            one_order_p=data_p.loc[(data_p['wavelength(nm)']<=order_w[n_order][1])&(data_p['wavelength(nm)']>=order_w[n_order][0])]

            if sky == 'Uniform':
                one_order_k=data_sky.loc[(data_sky['wavelength(nm)']<=order_w[n_order][1])&(data_sky['wavelength(nm)']>=order_w[n_order][0])]

            #central_pix=int(np.ceil(self.focal_plane.shape[0]/2))

            #==flux at the central pixel of PSF wings==

            focal_plane[central_pix, 0:len(one_order_s)]=one_order_s['dat_diff_0']
            print(one_order_s['dat_diff_0'][0:10])
            focal_plane[central_pix+d, 0:len(one_order_p)]=one_order_p['dat_diff_0']
            print(one_order_p['dat_diff_0'][0:10])
            
            if data_disk != None:
                one_order_ds=data_disk.loc[(data_disk['wavelength(nm)']<=order_w[n_order][1])&(data_disk['wavelength(nm)']>=order_w[n_order][0])]

                if disk_structure == 'two-side ring':
                    focal_plane[central_pix+disk_separation[0], 0:len(one_order_ds)]=one_order_ds['dat_diff_0']
                    focal_plane[central_pix+disk_separation[1], 0:len(one_order_ds)]=one_order_ds['dat_diff_0']
                
                if disk_structure =='one-side ring':
                    focal_plane[central_pix+disk_separation, 0:len(one_order_ds)]=one_order_ds['dat_diff_0']
                
                if disk_structure =='veiling':

                    print ('Please add the veiling factor into the injected planet spectrum, or add an extinction value when generate the planetary signal.')

                    data_disk == None 

                    

                else:
                    print ("Please select the disk structure keywords supported by the pipeline: 'one-side' or 'two-side' ")


            #if sky == True:
            #focal_plane[central_pix, 0:len(one_order_k)]+=(one_order_k['dat_diff_0'])

            print("The center of planet's PSF profile is set at pixel %s"%(central_pix+d))

            #==Spread out following the PSF wings==

            for z in range(1, pixel_posi):

                #add stellar signal
                loc_up=central_pix+z
                loc_be=central_pix-z

                focal_plane[loc_up,0:len(one_order_s)]+=one_order_s['dat_diff_%s'%z]
                focal_plane[loc_be,0:len(one_order_s)]+=one_order_s['dat_diff_%s'%z]


                #add planet signal
                if d+z <= pixel_posi:
                    loc_up_p=central_pix+d+z
                    loc_be_p=central_pix+d-z
                    focal_plane[loc_up_p,0:len(one_order_p)]+=one_order_p['dat_diff_%s'%z]
                    focal_plane[loc_be_p,0:len(one_order_p)]+=one_order_p['dat_diff_%s'%z]

                #add disk signal
                if disk_structure == 'one-side ring':
                    if disk_separation+z <= pixel_posi:
                        loc_up_ds=central_pix+disk_separation+z
                        loc_be_ds=central_pix+disk_separation-z
                        focal_plane[loc_up_ds,0:len(one_order_ds)]+=one_order_ds['dat_diff_%s'%z]
                        focal_plane[loc_be_ds,0:len(one_order_ds)]+=one_order_ds['dat_diff_%s'%z]   

                elif disk_structure =='two-side ring':  
                    #Inner disk               
                    if disk_separation[0]+z <= pixel_posi:
                        loc_up_ds=central_pix+disk_separation[0]+z
                        loc_be_ds=central_pix+disk_separation[0]-z
                        focal_plane[loc_up_ds,0:len(one_order_ds)]+=one_order_ds['dat_diff_%s'%z]
                        focal_plane[loc_be_ds,0:len(one_order_ds)]+=one_order_ds['dat_diff_%s'%z]
                    #Outer disk
                    if disk_separation[1]+z <= pixel_posi:
                        loc_up_ds=central_pix+disk_separation[1]+z
                        loc_be_ds=central_pix+disk_separation[1]-z
                        focal_plane[loc_up_ds,0:len(one_order_ds)]+=one_order_ds['dat_diff_%s'%z]
                        focal_plane[loc_be_ds,0:len(one_order_ds)]+=one_order_ds['dat_diff_%s'%z]


            #Add sky background
            #Uniform sky: Generalize the sky background to all pixels
            if sky == 'Uniform':
                #add sky signal
                one_order_k_re=np.tile(one_order_k['dat_diff_0'], (focal_plane.shape[0], 1))
                focal_plane[:, 0:len(one_order_k)]+=(one_order_k_re)


            #Add noise
            if noise == True and NDIT != None:

                N=NDIT
                if N > 50: 
                    print ("WARNING: TOO many exposures (>50) may crush the kernel")

                #add noise

                #Photon noise: lambda=photon counts for each pxiel
                if np.isnan(focal_plane).any() == True:
                    print ('WARNING: The inital 2D spectrum contains nan values. They will be simply masked for generation of photon noise distribution later on. ')
                    focal_plane[np.isnan(focal_plane)]=1e-10
                
                if np.array(focal_plane<0).any() == True:
                    print ('WARNING: The inital 2D spectrum contains negative values. They will be simply scaled to +1e-10 for generation of photon noise distribution later on. ')
                    focal_plane[focal_plane<0]=1e-10


                Photon_series=np.array([np.random.poisson(lam=focal_plane) for _ in range(N)])

                #Ron and Dark
                Dark_series=np.array([np.random.poisson(lam=dark, size=focal_plane.shape) for _ in range(N)])
                #Readout_series=np.array([np.random.poisson(lam=ron, size=focal_plane.shape) for _ in range(N)])
                #Using Gaussian distribution to simulate read-out-noise
                Readout_series=np.array([np.random.normal(loc=0.0, scale=ron, size=focal_plane.shape) for _ in range(N)])

                signal_tot[n_order]=np.add(Photon_series, Dark_series, Readout_series)

                #signal_stack[n_order]=np.mean(signal_tot[n_order], axis=0)

            elif noise == False and NDIT != None:

                N=NDIT

                Photon_series=np.array([focal_plane for _ in range(N)])

                signal_tot[n_order]=Photon_series

                signal_stack[n_order]=np.mean(signal_tot[n_order], axis=0)

            elif noise == False and NDIT == None:

                signal_tot[n_order] = focal_plane

            else:

                print ('NDIT cannot be none if noise added.')




        if plot_combination == True:

            width = len(self.aperture_list)

            fig, axs=plt.subplots(row,1,dpi=200, sharex=True, sharey=True)

            for n_order in range(0, row):

                ceil=np.percentile(signal_stack[n_order], ceil_percentage)
                nor = ImageNormalize(signal_stack[n_order], interval= ZScaleInterval(contrast=0.2), vmax=ceil,vmin=np.min(focal_plane))

                ff=axs[n_order]
                #if noise==True:
                ff.imshow(signal_stack[n_order], norm=nor ,cmap='Greys_r')
                fig.colorbar(ff.imshow(signal_stack[n_order], norm=nor, cmap='Greys_r'), ax=axs[n_order], shrink=0.3, aspect=50, \
                             location='bottom',pad=0.2)
                #else:
                #ff.imshow(signal_tot[n_order], norm=nor ,cmap='Greys_r')
                #fig.colorbar(ff.imshow(signal_tot[n_order], norm=nor, cmap='Greys_r'), ax=axs[n_order], shrink=0.3, aspect=50, \
                #location='bottom',pad=0.2)

                ff.set_title('order%s: %s nm - %s nm mean_stacking'%(np.max(self.order_num)-n_order,order_w[n_order][0],order_w[n_order][1]),fontsize=5)


                ff.hlines(y=central_pix+d,xmin=0,xmax=signal_stack[n_order].shape[1],ls='--',colors='gray', linewidth=0.8)
                ff.axis('off')
                ff.set_aspect(8)
                ff.set_ylim(central_pix-width,central_pix+width)

            fig.subplots_adjust(hspace=1.3)
            fig.suptitle('focal plane (%s*%s)\n$e^-$/pix/exposure'%(signal_stack[n_order].shape[0], signal_stack[n_order].shape[1]),\
                         x=0.5,fontsize=9)

            plt.close()

            return (signal_tot, fig)
        
        else: 
            return (signal_tot)


#-----------------------------------------
#Function for loading .json files

def load_json(path, permit):
    with open(path, '%s'%permit) as f:
        data=json.load(f)

    return (data)
