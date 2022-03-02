from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader

from astropy import units

import sys 
import os

from ossos.gui import logger

storing_directory = '/arc/projects/uvickbos/ML-MOD/new_cutouts/'


def pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p14.measure3.cands.astrom', 
        filename='15AP+0+0_p14.measure3.cands.astrom', real_exists=0):
    '''
    
    '''
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'

    parser = astrom.AstromParser()
    dlm = downloader.ImageCutoutDownloader()
    sources = parser.parse(full_filename)

    if real_exists:
        # make a list of loc to look at 
        # HERE
        real_full_filename = full_filename.replace('.cands.astrom', '.reals.astrom') 
        real_sources = parser.parse(real_full_filename)

        real_cands = []
        for real_source in real_sources.get_sources(): 
            print(real_source)

            for i,real_reading in enumerate(real_source.get_readings()):
                print(real_reading.dec)
                real_cands.append(real_reading.dec)


        cand=0
        for source in sources.get_sources(): 
            print(source)

            file_dir = filename + '_' + str(cand) + '/'
            sub_dir = storing_directory + file_dir
            os.mkdir(sub_dir)
            cand+=1
            for i,reading in enumerate(source.get_readings()):
                print(reading)

                if reading.dec in real_cands: # check up one earlier in source? always all 3 only
                    label=1
                else:
                    label=0

                cutout = dlm.download_cutout(reading, needs_apcor=True)
                cutout.hdulist.writeto(sub_dir + filename + '_' + str(i) + '_label=' + str(label) + '.fits', overwrite=True)

    else:
        label=0
        cand=0
        for source in sources.get_sources(): 
            print(source)
            file_dir = filename + '_' + str(cand) + '/'
            sub_dir = storing_directory + file_dir
            os.mkdir(sub_dir)
            cand+=1
            for i,reading in enumerate(source.get_readings()):
                print(reading)
                cutout = dlm.download_cutout(reading, needs_apcor=True)
                cutout.hdulist.writeto(sub_dir + filename + '_' + str(i) + '_label=' + str(label) + '.fits', overwrite=True)