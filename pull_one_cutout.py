from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader

from astropy import units

import sys 
import os

from ossos.gui import logger

storing_directory = '/arc/projects/uvickbos/ML-MOD/cutouts_debug/'


def pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p14.measure3.cands.astrom', 
        filename='15AP+0+0_p14.measure3.cands.astrom', real_exists=0):
    '''
    
    '''
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'

    parser = astrom.AstromParser()

    dlm = downloader.ImageCutoutDownloader()

    sources = parser.parse(full_filename) #full_filename)
    print(full_filename)

    file_dir = filename + '/'
    sub_dir = storing_directory + file_dir
    os.mkdir(sub_dir)

    for source in sources.get_sources()[1:2]:
        for i,reading in enumerate(source.get_readings()):
            cutout = dlm.download_cutout(reading, needs_apcor=True)
            cutout.hdulist.writeto(sub_dir + filename + str(i) + '.fits', overwrite=True)
            mjd_obs = float(cutout.fits_header.get('MJD-OBS'))
            exptime = float(cutout.fits_header.get('EXPTIME'))



pull_cutout()