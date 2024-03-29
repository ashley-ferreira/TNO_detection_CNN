from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader

from astropy import units

import sys 
import os

from ossos.gui import logger

storing_directory = '/arc/projects/uvickbos/ML-MOD/triplets_wTNO_examples/'


def pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p34.measure3.cands.astrom', 
        filename='15AP+0+0_p34.measure3.cands.astrom', real_exists=0):
    '''
    
    '''
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'

    parser = astrom.AstromParser()

    dlm = downloader.ImageCutoutDownloader()

    sources = parser.parse(full_filename) #full_filename)
    print(full_filename)

    file_dir = filename + 'new3/'
    sub_dir = storing_directory + file_dir
    os.mkdir(sub_dir)#, exist_ok=True)

    cand=0
    for source in sources.get_sources():
        print(source)
        cand+=1
        for i,reading in enumerate(source.get_readings()):
            print(reading)
            cutout = dlm.download_cutout(reading, needs_apcor=True)
            cutout.hdulist.writeto(sub_dir + filename + '_' + str(cand) + '_' + str(i) + '.fits', overwrite=True)
            mjd_obs = float(cutout.fits_header.get('MJD-OBS'))
            exptime = float(cutout.fits_header.get('EXPTIME'))



#pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p28.measure3.cands.astrom', filename='15AP+0+0_p28.measure3.cands.astrom')
#pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p24.measure3.cands.astrom', filename='15AP+0+0_p24.measure3.cands.astrom')
#pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p17.measure3.cands.astrom', filename='15AP+0+0_p17.measure3.cands.astrom')
#pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p15.measure3.cands.astrom', filename='15AP+0+0_p15.measure3.cands.astrom')
#pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP-2-2/15AP-2-2_p1.measure3.cands.astrom', filename='15AP-2-2_p1.measure3.cands.astrom')
pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+0+0/15AP+0+0_p14.measure3.cands.astrom', filename='15AP+0+0_p14.measure3.cands.astrom')