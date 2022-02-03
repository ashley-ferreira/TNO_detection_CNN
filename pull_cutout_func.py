from ossos import storage
from ossos import astrom
from ossos.downloads.cutouts import downloader
import sys 

def pull_cutout(filename):
    '''
    
    '''
    storage.DBIMAGES = 'vos:OSSOS/dbimages'
    storage.MEASURE3 = 'vos:OSSOS/measure3'

    parser = astrom.AstromParser()

    dlm = downloader.ImageCutoutDownloader(slice_rows=200, slice

    sources = parser.parse(filename)

    for source in sources.get_sources()[1:2]:
        for i,reading in enumerate(source.get_readings()):
            print(reading)
            cutout = dlm.download_cutout(reading, needs_apcor=True)
            print(cutout.hdulist)
            cutout.hdulist.writeto('~/TNO_detection/'+str(i)+'.fits', overwrite=True)
            mjd_obs = float(cutouts.fits_header.get('MJD-OBS'))
            exptime = float(cutouts.fits_header.get('EXPTIME'))
            print(f'Exposure taken at: {mjd_obs} with exposure time {exptime}')



# loop through folder instead?