from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader

from astropy import units
cutout_size = 140/4 # pixels as radius

import sys 

from ossos.gui import logger

logger.set_debug()


def pull_cutout(full_filename='vos:OSSOS/measure3/2015A-P/15AP+2-1/15AP+2-1_p36.measure3.cands.astrom', 
        filename='15AP+2-1_p36.measure3.cands.astrom', real_exists=0):
    '''
    
    '''
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'

    parser = astrom.AstromParser()

    dlm = downloader.ImageCutoutDownloader()

    sources = parser.parse(full_filename)

    for source in sources.get_sources()[1:2]:
        for i,reading in enumerate(source.get_readings()):
            reading.uncertainty_ellipse.a = cutout_size*0.185/2/2.5 * units.arcsecond
            print(reading)
            print(reading.uncertainty_ellipse)
            cutout = dlm.download_cutout(reading, needs_apcor=True)
            print(cutout.hdulist)
            cutout.hdulist.writeto('/arc/projects/uvickbos/ML-MOD/140_pix_cutouts/'+ filename + str(i) + '_label=' + str(real_exists) + '.fits', overwrite=True)
            mjd_obs = float(cutout.fits_header.get('MJD-OBS'))
            exptime = float(cutout.fits_header.get('EXPTIME'))
            print(f'Exposure taken at: {mjd_obs} with exposure time {exptime}')
